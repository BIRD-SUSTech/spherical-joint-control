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
            estimated_cable_lengths=self._cable_lengths_from_angles(angles),
            estimated_joint_angles=self._joint_angles_from_lengths(angles),
        )
        try:
            self._output_queue.put_nowait(state)
        except queue.Full:
            pass

    def emergency_stop(self) -> None:
        logger.info("LogServoController: emergency stop (no-op)")

    @staticmethod
    def _cable_lengths_from_angles(angles: List[float]) -> List[float]:
        return [0.0] * len(angles)

    @staticmethod
    def _joint_angles_from_lengths(lengths: List[float]) -> List[float]:
        return [0.0, 0.0]


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

        while wp_index < len(waypoints) and not self._stop_event.is_set():
            wp = waypoints[wp_index]
            deadline = time.perf_counter() + wp.duration_s
            logger.debug("Waypoint %d/%d: pitch=%.1f, yaw=%.1f, dur=%.1fs",
                         wp_index + 1, len(waypoints),
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

        logger.info("PIDServoController: trajectory complete")

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
            estimated_cable_lengths=[target_pitch, target_yaw, current_pitch, current_yaw],
            estimated_joint_angles=[target_pitch, target_yaw],
        )
        try:
            self._output_queue.put_nowait(state)
        except queue.Full:
            pass
