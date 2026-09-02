"""舵机控制模块。

提供三层控制能力：
  - ServoController (ABC): 多舵机抽象接口
  - LogServoController:      无硬件日志桩
  - PIDServoController:      基于 mocap 反馈的 PID 闭环轨迹跟踪
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np
import serial
from numpy.typing import NDArray

from control_model.base_PID.pid_controller import BasePIDController
from control_model.excitation import sample_segment

from .config import ServoConfig
from .utils.data_types import ServoState
from .utils.trajectory import Trajectory

logger = logging.getLogger(__name__)

PoseCallback = Callable[[], tuple[float, float]]


# ==========================================================================
# 抽象基类
# ==========================================================================

class ServoController(ABC):
    """多舵机控制器抽象接口。"""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_angles(self, angles: List[float]) -> None: ...

    def emergency_stop(self) -> None:
        """紧急停止，默认空实现。"""

    def move_trajectory(
        self,
        trajectory: List[List[float]],
        interval_s: float,
        stop_event: threading.Event | None = None,
    ) -> None:
        for waypoint in trajectory:
            if stop_event is not None and stop_event.is_set():
                break
            self.set_angles(waypoint)
            time.sleep(interval_s)


# ==========================================================================
# 无硬件日志桩
# ==========================================================================

class LogServoController(ServoController):
    """无硬件桩：仅将目标角度写入输出队列。"""

    def __init__(
        self,
        config: ServoConfig,
        output_queue: queue.Queue,
    ):
        self._config = config
        self._output_queue = output_queue
        self._connected = False

    def connect(self) -> bool:
        logger.info("LogServoController: logging mode (no hardware)")
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def set_angles(self, angles: List[float]) -> None:
        if not self._connected:
            return
        state = ServoState(
            pc_timestamp_ns=time.perf_counter_ns(),
            pc_receive_unix_time_ms=int(time.time() * 1000),
            target_angles=list(angles),
        )
        try:
            self._output_queue.put_nowait(state)
        except queue.Full:
            pass

    def emergency_stop(self) -> None:
        logger.info("LogServoController: emergency stop (no-op)")


# ==========================================================================
# 串口舵机驱动
# ==========================================================================

class SerialServoDriver(ServoController):
    """串口舵机驱动：向指定串口发送四路舵机角度。

    输入为归一化舵机指令 [-1, 1]，发送前映射为舵机角度 (度):
        deg = norm * 135   →   [-135, 135]  (四舍五入为整数)

    输出协议: 四个逗号分隔的整数 (度)，以 \\n 结尾。
    例如: "68,-68,122,0\\n"

    用法:
        driver = SerialServoDriver(port="COM3", baudrate=115200)
        driver.connect()
        driver.send(np.array([0.5, -0.5, 0.9, 0.0]))
        ...
        driver.emergency_stop()
        driver.disconnect()
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.5,
        half_range_deg: float = 135.0,
    ):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._half_range_deg = half_range_deg
        self._ser: Optional[serial.Serial] = None

    def connect(self) -> bool:
        if self._ser and self._ser.is_open:
            return True
        try:
            self._ser = serial.Serial(
                port=self._port, baudrate=self._baudrate, timeout=self._timeout
            )
        except serial.SerialException:
            logger.exception("SerialServoDriver: 无法打开串口 %s", self._port)
            self._ser = None
            return False
        logger.info("SerialServoDriver: 已打开 %s @ %d baud", self._port, self._baudrate)
        return True

    def disconnect(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def set_angles(self, angles: List[float]) -> None:
        """ServoController 接口：接受归一化舵机角度列表 [-1, 1]."""
        self.send(np.asarray(angles, dtype=float))

    def send(self, norm: NDArray) -> None:
        """发送一帧舵机指令 (shape (4,) 归一化 [-1,1] → 整数度 [-135,135])."""
        if self._ser is None or not self._ser.is_open:
            return
        line = ",".join(
            f"{np.clip(v, -1.0, 1.0) * self._half_range_deg:.0f}"
            for v in norm
        ) + "\n"
        self._ser.write(line.encode("ascii"))

    def emergency_stop(self) -> None:
        logger.info("SerialServoDriver: 紧急停止 (发送零指令)")
        self.send(np.zeros(4))


# ==========================================================================
# PID 闭环轨迹跟踪控制器
# ==========================================================================

class PIDServoController:
    """PID 闭环轨迹跟踪控制器。

    用法:
        ctrl = PIDServoController(
            pid=BasePIDController(),
            trajectory=traj,
            pose_source=orchestrator.get_current_pose,
            command_sink=lambda norm: hardware.send(norm),
            output_queue=q_servo,
        )
        ctrl.connect()
        ctrl.start()
        ...
        ctrl.stop()
        ctrl.disconnect()
    """

    LOOP_PERIOD_S: float = 0.005   # 控制回路更新周期 (s), ~200Hz
    UPDATE_DT_MIN: float = 0.001   # PID dt 下限，防除零

    def __init__(
        self,
        pid: BasePIDController,
        trajectory: Trajectory,
        pose_source: PoseCallback,
        command_sink: Callable[[NDArray], None] | None,
        output_queue: queue.Queue,
    ):
        self._pid = pid
        self._trajectory = trajectory
        self._pose_source = pose_source
        self._command_sink = command_sink
        self._output_queue = output_queue

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        logger.info("PIDServoController: ready (pid gains: pitch Kp=%.2f, yaw Kp=%.2f)",
                    self._pid.pitch_pid._g.Kp, self._pid.yaw_pid._g.Kp)
        return True

    def disconnect(self) -> None:
        self.stop()

    def start(self) -> None:
        """启动控制线程。"""
        if self._thread and self._thread.is_alive():
            logger.warning("PIDServoController: already running")
            return
        self._stop_event.clear()
        self._pid.reset()
        self._thread = threading.Thread(
            target=self._control_loop, name="PIDServo", daemon=True
        )
        self._thread.start()
        logger.info("PIDServoController: control thread started, %d waypoints",
                    len(self._trajectory))

    def stop(self) -> None:
        """停止控制线程，零位回中。"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._command_sink:
            try:
                self._command_sink(np.zeros(4))
            except Exception:
                pass

    def emergency_stop(self) -> None:
        """紧急停止，立即回中。"""
        logger.warning("PIDServoController: EMERGENCY STOP")
        self._stop_event.set()
        if self._command_sink:
            try:
                self._command_sink(np.zeros(4))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 控制回路
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        last_time = time.perf_counter()
        wp_index = 0
        waypoints = self._trajectory.waypoints
        total_wp = len(waypoints)
        loop_count = 0

        while not self._stop_event.is_set():
            wp = waypoints[wp_index]
            deadline = time.perf_counter() + wp.duration_s
            logger.debug("Waypoint %d/%d (loop %d): pitch=%.1f, yaw=%.1f, dur=%.1fs",
                         wp_index + 1, total_wp, loop_count,
                         wp.pitch_deg, wp.yaw_deg, wp.duration_s)

            while time.perf_counter() < deadline and not self._stop_event.is_set():
                now = time.perf_counter()
                dt = max(now - last_time, self.UPDATE_DT_MIN)
                last_time = now

                curr_pitch, curr_yaw = self._pose_source()

                servo_norm = self._pid.update(
                    wp.pitch_deg, wp.yaw_deg,
                    curr_pitch, curr_yaw,
                    dt,
                )

                if self._command_sink:
                    try:
                        self._command_sink(servo_norm)
                    except Exception:
                        logger.exception("Hardware command failed")

                self._log_servo_state(servo_norm, wp.pitch_deg, wp.yaw_deg,
                                      curr_pitch, curr_yaw)

                time.sleep(self.LOOP_PERIOD_S)

            wp_index += 1
            if wp_index >= total_wp:
                wp_index = 0
                loop_count += 1
                logger.info("PIDServoController: loop %d complete, restarting", loop_count)

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def _log_servo_state(
        self,
        servo_norm: NDArray,
        target_pitch: float,
        target_yaw: float,
        current_pitch: float,
        current_yaw: float,
    ) -> None:
        now_ns = time.perf_counter_ns()
        now_ms = int(time.time() * 1000)
        state = ServoState(
            pc_timestamp_ns=now_ns,
            pc_receive_unix_time_ms=now_ms,
            target_angles=list(servo_norm),
            target_pitch=target_pitch,
            target_yaw=target_yaw,
            current_pitch=current_pitch,
            current_yaw=current_yaw,
        )
        try:
            self._output_queue.put_nowait(state)
        except queue.Full:
            pass


# ==========================================================================
# 开环激励控制器（无 IK，直接在差分/共模空间驱动舵机）
# ==========================================================================

class OpenLoopExcitationController:
    """开环激励控制器：大摆幅平滑激励，直接驱动舵机，采集 I/O 数据。

    激励在差分/共模空间生成 (d1, d2, p) -> 4 路归一化舵机角，不使用几何 IK。
    适合"IK 不可信、先采集开环数据辨识系统"的场景。

    用法:
        ctrl = OpenLoopExcitationController(
            segments=segments, pretension_norm=0.15,
            command_sink=driver.send, output_queue=q_servo,
            pose_source=orchestrator.get_current_pose, fs=100.0,
        )
        ctrl.connect(); ctrl.start(); ... ctrl.stop()
    """

    def __init__(
        self,
        segments: list[dict],
        pretension_norm: float,
        command_sink: Callable[[NDArray], None] | None,
        output_queue: queue.Queue,
        pose_source: PoseCallback | None = None,
        fs: float = 100.0,
        safety_limit_deg: float = 55.0,
        calib_amp: float = 0.7,
        inter_segment_dwell_s: float = 1.0,
    ):
        self._segments = segments
        self._p = pretension_norm
        self._command_sink = command_sink
        self._output_queue = output_queue
        self._pose_source = pose_source
        self._fs = max(fs, 1.0)
        self._safety_limit_deg = safety_limit_deg
        self._calib_amp = calib_amp
        self._inter_dwell_s = max(inter_segment_dwell_s, 0.0)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        logger.info("OpenLoopExcitationController: ready (%d segments, pretension=%.2f)",
                    len(self._segments), self._p)
        return True

    def disconnect(self) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("OpenLoopExcitationController: already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="OpenLoopExcitation", daemon=True
        )
        self._thread.start()
        logger.info("OpenLoopExcitationController: thread started, %d segments",
                    len(self._segments))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._send_zero()

    def emergency_stop(self) -> None:
        logger.warning("OpenLoopExcitationController: EMERGENCY STOP")
        self._stop_event.set()
        self._send_zero()

    @property
    def total_duration_s(self) -> float:
        """完整 schedule 总时长（各段 + 段间中立停留）。"""
        seg_total = sum(float(seg.get("duration_s", 0.0)) for seg in self._segments)
        return seg_total + len(self._segments) * self._inter_dwell_s

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _send_zero(self) -> None:
        if self._command_sink:
            try:
                self._command_sink(np.zeros(4))
            except Exception:
                pass

    def _send_neutral(self) -> None:
        """发送中立（零差分、只留共模预紧）。"""
        if self._command_sink:
            try:
                self._command_sink(np.full(4, self._p))
            except Exception:
                pass

    def _send(self, cmd: NDArray) -> None:
        if self._command_sink:
            try:
                self._command_sink(cmd)
            except Exception:
                logger.exception("Open-loop command failed")

    def _current_pose(self) -> tuple[float, float]:
        if self._pose_source:
            try:
                return self._pose_source()
            except Exception:
                pass
        return (0.0, 0.0)

    def _log(self, cmd: NDArray, segment_id: int = -1) -> None:
        pitch, yaw = self._current_pose()
        state = ServoState(
            pc_timestamp_ns=time.perf_counter_ns(),
            pc_receive_unix_time_ms=int(time.time() * 1000),
            target_angles=list(cmd),
            target_pitch=0.0,
            target_yaw=0.0,
            current_pitch=pitch,
            current_yaw=yaw,
            segment_id=segment_id,
        )
        try:
            self._output_queue.put_nowait(state)
        except queue.Full:
            pass

    def _check_safety(self) -> bool:
        if self._pose_source is None or self._safety_limit_deg <= 0:
            return True
        pitch, yaw = self._current_pose()
        if abs(pitch) > self._safety_limit_deg or abs(yaw) > self._safety_limit_deg:
            logger.error("Safety limit exceeded: pitch=%.1f yaw=%.1f -> stop",
                         pitch, yaw)
            return False
        return True

    def run_calibration(self, duration_s: float) -> None:
        """CALIBRATION 段：慢速大摆幅 Lissajous，驱动关节覆盖工作空间。

        供 IMU↔动捕四元数线性对齐使用（需要两轴都有足够运动）。
        阻塞执行 duration_s，结束后回到中立预紧。
        """
        if duration_s <= 0:
            return
        seg = {"kind": "lissajous", "amp": self._calib_amp,
               "f1": 0.05, "f2": 0.08, "duration_s": duration_s}
        t, u = sample_segment(seg, self._p, self._fs)
        dt = 1.0 / self._fs
        logger.info("Open-loop calibration: amp=%.2f, %.1fs (%d samples)",
                    self._calib_amp, duration_s, len(t))
        for i in range(len(t)):
            if self._stop_event.is_set():
                break
            cmd = u[:, i]
            self._send(cmd)
            self._log(cmd)
            if not self._check_safety():
                self._stop_event.set()
                self._send_neutral()
                return
            time.sleep(dt)
        self._send_neutral()

    def _loop(self) -> None:
        dt = 1.0 / self._fs
        for seg_idx, seg in enumerate(self._segments):
            if self._stop_event.is_set():
                break
            t, u = sample_segment(seg, self._p, self._fs)
            logger.info("Open-loop segment %d/%d: %s (amp=%.2f, %d samples)",
                        seg_idx + 1, len(self._segments), seg.get("kind"),
                        seg.get("amp", float("nan")), len(t))
            for i in range(len(t)):
                if self._stop_event.is_set():
                    break
                cmd = u[:, i]
                self._send(cmd)
                self._log(cmd, segment_id=seg_idx)
                if not self._check_safety():
                    self._stop_event.set()
                    self._send_zero()
                    return
                time.sleep(dt)
            # 段间回到中立（只留共模预紧、零差分），停留 dwell_s 让系统稳定并给数据分段
            if not self._stop_event.is_set():
                neutral = np.full(4, self._p)
                for _ in range(int(self._inter_dwell_s / dt)):
                    if self._stop_event.is_set():
                        break
                    self._send(neutral)
                    self._log(neutral, segment_id=-1)
                    time.sleep(dt)
