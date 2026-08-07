"""总协调器：生命周期管理、多阶段采集协调、数据汇入和 CSV 输出。"""

from __future__ import annotations

import json
import logging
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from control_model.base_PID.pid_controller import BasePIDController
from control_model.open_loop.open_loop_controller import OpenLoopController
from .config import Config, OrchestratorConfig, OutputConfig
from .servo_controller import (
    LogServoController,
    PIDServoController,
    SerialServoDriver,
    ServoController,
)
from .utils.csv_writer import CsvWriter
from .utils.data_types import (
    MOCAP_CSV_COLUMNS,
    SERVO_CSV_COLUMNS,
    FORCE_CSV_COLUMNS,
    IMU_CSV_COLUMNS,
    CollectionPhase,
    ForceData,
    ImuPacket,
    MocapFrame,
    RigidBody,
    ServoState,
)
from .sensor_collectors.force_collector import ForceCollector
from .sensor_collectors.imu_collector import ImuCollector
from .sensor_collectors.mocap_collector import MocapCollector
from .utils.session import SessionManager, SessionPaths
from .utils.trajectory import Trajectory, make_circle_trajectory, make_sine_trajectory

logger = logging.getLogger(__name__)


class Orchestrator:
    """统一数据采集协调器。"""

    def __init__(self, config: Config):
        self._config = config
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._phase: CollectionPhase = CollectionPhase.STATIC
        self._phase_lock = threading.Lock()

        # 输出队列
        queue_max = config.output.queue_maxsize
        self._q_mocap = queue.Queue(maxsize=queue_max)
        self._q_imu = queue.Queue(maxsize=queue_max)
        self._q_force = queue.Queue(maxsize=queue_max)
        self._q_servo = queue.Queue(maxsize=queue_max)

        # 采集器与控制器（构造时不连接硬件）
        self._mocap: Optional[MocapCollector] = None
        self._imu: Optional[ImuCollector] = None
        self._force: Optional[ForceCollector] = None
        self._servo: Optional[ServoController] = None

        # PID 闭环跟踪
        self._pid_servo: Optional[PIDServoController] = None
        self._servo_driver: Optional[SerialServoDriver] = None
        self._open_loop: Optional[OpenLoopController] = None
        self._pose_lock = threading.Lock()
        self._current_pitch_deg = 0.0
        self._current_yaw_deg = 0.0
        self._ref_quat: Optional[NDArray] = None  # 中立位参考四元数

        # CSV 写入器
        self._writers: dict[str, Optional[CsvWriter]] = {}
        self._session_paths: Optional[SessionPaths] = None

        # 统计
        self._servo_dropped = 0

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def run(self) -> int:
        """运行采集会话。返回 0 表示正常完成。"""
        self._setup_signal_handlers()

        # 1. 创建会话目录
        session_mgr = SessionManager(
            root_dir=self._config.output.root_dir,
            prefix=self._config.output.session_prefix,
        )
        self._session_paths = session_mgr.create_session()
        logger.info("Session directory: %s", self._session_paths.dir)

        # 2. 初始化 CSV 写入器
        self._setup_csv_writers()

        # 3. 初始化采集器
        if not self._setup_collectors():
            logger.error("Failed to initialize collectors; aborting")
            self._cleanup()
            return 1

        # 4. 连接硬件
        self._connect_all()

        # 5. 启动采集线程
        self._start_collectors()

        # 6. 启动 consumer 线程 — 每个传感器独立写 CSV，并行驶入 NAS
        consumer_threads: list[threading.Thread] = []
        for name, queue_obj, writer in [
            ("mocap", self._q_mocap, self._writers.get("mocap")),
            ("imu",   self._q_imu,   self._writers.get("imu")),
            ("force", self._q_force, self._writers.get("force")),
            ("servo", self._q_servo, self._writers.get("servo")),
        ]:
            if writer is None:
                continue
            t = threading.Thread(
                target=self._single_consumer,
                args=(name, queue_obj, writer),
                name=f"Consumer-{name}",
            )
            t.start()
            consumer_threads.append(t)
        logger.info("%d consumer threads started", len(consumer_threads))

        # 7. 阶段管理：静止段 → 标定段 → 探索段
        self._run_phases()

        # 8. 停止采集器 → consumers 排空队列后退出
        logger.info("Stopping collectors...")
        self._stop_event.set()
        self._stop_collectors()
        for t in consumer_threads:
            t.join()
        logger.info("All consumer threads stopped")

        # 9. 落盘并关闭 CSV
        self._cleanup()

        # 11. 写元数据
        self._write_metadata()
        logger.info(
            "Session complete. Data saved to %s", self._session_paths.dir
        )
        return 0

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            logger.info("Received signal %d, stopping...", signum)
            self._stop_event.set()

        try:
            signal.signal(signal.SIGINT, handler)
        except ValueError:
            pass

    def _setup_csv_writers(self) -> None:
        out = self._config.output
        sp = self._session_paths

        mocap_start = threading.Event()
        mocap_start.set()

        self._writers = {
            "mocap": CsvWriter(sp.mocap_csv, MOCAP_CSV_COLUMNS, flush_interval=out.flush_interval_rows)
            if out.enable_mocap_csv else None,
            "imu": CsvWriter(sp.imu_csv, IMU_CSV_COLUMNS, flush_interval=out.flush_interval_rows)
            if out.enable_imu_csv else None,
            "force": CsvWriter(sp.force_csv, FORCE_CSV_COLUMNS, flush_interval=out.flush_interval_rows)
            if out.enable_force_csv else None,
            "servo": CsvWriter(sp.servo_csv, SERVO_CSV_COLUMNS, flush_interval=out.flush_interval_rows)
            if out.enable_servo_csv else None,
        }
        for w in self._writers.values():
            if w:
                w.open()

    def _setup_collectors(self) -> bool:
        cfg = self._config
        if cfg.output.enable_mocap_csv:
            try:
                self._mocap = MocapCollector(
                    cfg.mocap, self._q_mocap, self._start_event, self._stop_event,
                    pose_callback=self._update_pose_from_mocap,
                )
            except Exception:
                logger.exception("Failed to create MocapCollector (SDK not installed?)")
                self._mocap = None

        if cfg.output.enable_imu_csv:
            self._imu = ImuCollector(
                cfg.imu, self._q_imu, self._start_event, self._stop_event
            )

        if cfg.output.enable_force_csv:
            self._force = ForceCollector(
                cfg.force, self._q_force, self._start_event, self._stop_event
            )

        if self._should_use_pid_servo():
            self._setup_pid_servo()
        elif cfg.output.enable_servo_csv:
            self._servo = LogServoController(cfg.servo, self._q_servo)

        return True

    def _should_use_pid_servo(self) -> bool:
        cfg = self._config.servo
        if cfg.trajectory_type == "waypoints":
            return bool(cfg.trajectory_waypoints)
        return cfg.trajectory_type in ("circle", "sine")

    def _setup_pid_servo(self) -> None:
        cfg = self._config.servo
        trajectory = self._make_trajectory(cfg)
        pid = BasePIDController(
            pretension_mm=cfg.pretension_mm,
            filter_tau_s=cfg.mocap_filter_tau_s,
            pitch_kp=cfg.pitch_kp,
            pitch_ki=cfg.pitch_ki,
            yaw_kp=cfg.yaw_kp,
            yaw_ki=cfg.yaw_ki,
            integral_max=cfg.integral_max,
        )
        self._open_loop = OpenLoopController(pretension_mm=cfg.pretension_mm)

        driver: Optional[SerialServoDriver] = None
        if cfg.enabled and cfg.port:
            driver = SerialServoDriver(port=cfg.port, baudrate=cfg.baudrate)
        self._servo_driver = driver

        self._pid_servo = PIDServoController(
            pid=pid,
            trajectory=trajectory,
            pose_source=self._get_current_pose,
            command_sink=driver.send if driver else None,
            output_queue=self._q_servo,
        )
        mode = f"serial:{cfg.port}" if driver else "log-only"
        logger.info("PID servo configured: %d waypoints, type=%s (%s)",
                    len(trajectory), cfg.trajectory_type, mode)

    @staticmethod
    def _make_trajectory(cfg) -> Trajectory:
        if cfg.trajectory_type == "waypoints":
            return Trajectory.from_pairs(cfg.trajectory_waypoints)
        elif cfg.trajectory_type == "circle":
            return make_circle_trajectory(
                radius_deg=cfg.trajectory_amplitude_deg,
                period_s=cfg.trajectory_period_s,
                steps=cfg.trajectory_steps,
            )
        elif cfg.trajectory_type == "sine":
            return make_sine_trajectory(
                pitch_amplitude_deg=cfg.trajectory_amplitude_deg,
                yaw_amplitude_deg=cfg.trajectory_amplitude_deg,
                period_s=cfg.trajectory_period_s,
                steps=cfg.trajectory_steps,
            )
        else:
            return Trajectory([])

    def _connect_all(self) -> None:
        if self._mocap and not self._mocap.connect():
            logger.warning("Mocap connection failed; continuing without mocap")
            self._mocap = None
        if self._servo_driver and not self._servo_driver.connect():
            logger.warning("Servo serial open failed; continuing log-only")
        else:
            # 舵机初始化：发送零位保持中立
            if self._servo_driver:
                self._servo_driver.send(np.zeros(4))
                logger.info("Servo initialized to neutral position")
        if self._pid_servo:
            self._pid_servo.connect()
        elif self._servo:
            self._servo.connect()

    def _start_collectors(self) -> None:
        """启动所有采集线程，并等待就绪。"""
        if self._mocap:
            self._mocap.start()
        if self._imu:
            self._imu.start()
        if self._force:
            self._force.start()
        logger.info("All collectors started; waiting %ds for sensors to stabilize...",
                    self._config.orchestrator.static_duration_s)

    # ------------------------------------------------------------------
    # 阶段管理
    # ------------------------------------------------------------------

    def _run_phases(self) -> None:
        orch = self._config.orchestrator

        # 静止段
        logger.info("=== PHASE: STATIC (%.0fs) ===", orch.static_duration_s)
        logger.info("Keep the linkage STILL. IMU will calibrate zero point.")
        self._set_phase(CollectionPhase.STATIC)
        self._sleep_with_progress(orch.static_duration_s)

        # 标定段
        logger.info("=== PHASE: CALIBRATION (%.0fs) ===", orch.calibration_duration_s)
        self._set_phase(CollectionPhase.CALIBRATION)
        if self._servo_driver and self._open_loop:
            logger.info("Running open-loop calibration sweep...")
            self._run_calibration_sweep(orch.calibration_duration_s)
        else:
            logger.info(
                "Move the linkage in BOTH pitch and roll directions, "
                "at least +/- 15 deg in each axis."
            )
            self._sleep_with_progress(orch.calibration_duration_s)

        # 探索段
        if orch.exploration_duration_s is not None:
            logger.info("=== PHASE: EXPLORATION (%.0fs) ===", orch.exploration_duration_s)
        else:
            logger.info("=== PHASE: EXPLORATION (Ctrl+C to stop) ===")
        self._set_phase(CollectionPhase.EXPLORATION)

        if self._pid_servo:
            self._pid_servo.start()
        self._sleep_with_progress(orch.exploration_duration_s or float("inf"))
        if self._pid_servo:
            self._pid_servo.stop()

    def _set_phase(self, phase: CollectionPhase) -> None:
        with self._phase_lock:
            self._phase = phase

    def _current_phase(self) -> str:
        with self._phase_lock:
            return self._phase.value

    def _sleep_with_progress(self, duration_s: float) -> None:
        start = time.perf_counter()
        interval = self._config.orchestrator.progress_report_interval_s
        elapsed = 0.0
        while not self._stop_event.is_set():
            remaining = duration_s - elapsed
            if remaining <= 0:
                break
            sleep_time = min(interval, remaining)
            time.sleep(sleep_time)
            elapsed = time.perf_counter() - start

    def _run_calibration_sweep(self, duration_s: float) -> None:
        """在标定阶段运行开环正弦扫频轨迹."""
        amp = self._config.servo.calibration_amplitude_deg
        traj = make_sine_trajectory(
            pitch_amplitude_deg=amp,
            yaw_amplitude_deg=amp,
            period_s=duration_s,
            steps=max(20, int(duration_s * 10)),
        )
        logger.info("Calibration sweep: %d waypoints over %.1fs", len(traj), duration_s)

        for wp in traj:
            if self._stop_event.is_set():
                break
            servo_norm = self._open_loop.command(wp.pitch_deg, wp.yaw_deg)
            if self._servo_driver:
                self._servo_driver.send(servo_norm)
            self._log_servo_state(servo_norm, wp.pitch_deg, wp.yaw_deg,
                                  self._current_pitch_deg, self._current_yaw_deg)
            time.sleep(wp.duration_s)

        # 扫频结束回到中立位
        if self._servo_driver:
            self._servo_driver.send(np.zeros(4))
        logger.info("Calibration sweep complete")

    def _log_servo_state(
        self,
        servo_norm: NDArray,
        target_pitch: float,
        target_yaw: float,
        current_pitch: float,
        current_yaw: float,
    ) -> None:
        """写一条舵机状态到输出队列."""
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
            self._q_servo.put_nowait(state)
        except queue.Full:
            self._servo_dropped += 1

    # ------------------------------------------------------------------
    # Consumer 线程
    # ------------------------------------------------------------------

    def _single_consumer(
        self, name: str, q: queue.Queue, writer: CsvWriter,
    ) -> None:
        """单传感器 consumer：读队列 → 写 CSV，stop_event 后排空再退出。"""
        while not self._stop_event.is_set():
            try:
                item = q.get(timeout=0.1)
                self._write_item(writer, item)
            except queue.Empty:
                continue

        # 排空残余
        while True:
            try:
                item = q.get_nowait()
                self._write_item(writer, item)
            except queue.Empty:
                break
        logger.info("Consumer-%s stopped, %d rows written", name, writer.row_count)

    def _write_item(self, writer: CsvWriter, item) -> None:
        phase_str = self._current_phase()

        if isinstance(item, MocapFrame):
            # pose 已在 mocap 采集线程中通过回调更新，这里只写 CSV
            rows = item.to_csv_rows()
            for row in rows:
                row["phase"] = phase_str
                writer.write_row(row)
        elif isinstance(item, ImuPacket):
            row = item.to_csv_row()
            row["phase"] = phase_str
            writer.write_row(row)
        elif isinstance(item, ForceData):
            row = item.to_csv_row()
            row["phase"] = phase_str
            writer.write_row(row)
        elif isinstance(item, ServoState):
            row = item.to_csv_row()
            row["phase"] = phase_str
            writer.write_row(row)

    # ------------------------------------------------------------------
    # 停止与清理
    # ------------------------------------------------------------------

    def _stop_collectors(self) -> None:
        timeout = self._config.orchestrator.shutdown_timeout_s
        if self._mocap:
            self._mocap.stop()
            self._mocap.join(timeout=timeout)
        if self._imu:
            self._imu.stop()
            self._imu.join(timeout=timeout)
        if self._force:
            self._force.stop()
            self._force.join(timeout=timeout)
        if self._pid_servo:
            self._pid_servo.emergency_stop()
            self._pid_servo.disconnect()
            if self._servo_driver:
                self._servo_driver.disconnect()
        elif self._servo:
            self._servo.emergency_stop()
            self._servo.disconnect()

    def _cleanup(self) -> None:
        for w in self._writers.values():
            if w:
                w.close()

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------

    def _write_metadata(self) -> None:
        sp = self._session_paths
        row_counts = {}
        dropped = {}
        for name, writer in self._writers.items():
            if writer:
                row_counts[name] = writer.row_count
        if self._mocap:
            dropped["mocap"] = self._mocap.dropped_count
        if self._imu:
            dropped["imu"] = self._imu.dropped_count
            row_counts["imu_raw"] = self._imu.row_count
        if self._force:
            dropped["force"] = self._force.dropped_count
        dropped["servo"] = self._servo_dropped

        metadata = {
            "session_dir": str(sp.dir),
            "started_at": datetime.now().isoformat(),
            "row_counts": row_counts,
            "dropped_frames": dropped,
            "config_snapshot": {
                "mocap": _dataclass_dict(self._config.mocap),
                "imu": _dataclass_dict(self._config.imu),
                "force": _dataclass_dict(self._config.force),
                "servo": _dataclass_dict(self._config.servo),
                "output": _dataclass_dict(self._config.output),
                "orchestrator": _dataclass_dict(self._config.orchestrator),
            },
        }
        sp.metadata_json.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Metadata written to %s", sp.metadata_json)


    # ------------------------------------------------------------------
    # 姿态反馈
    # ------------------------------------------------------------------

    def _get_current_pose(self) -> tuple[float, float]:
        """供 PIDServoController 读取当前姿态 (线程安全)."""
        with self._pose_lock:
            return (self._current_pitch_deg, self._current_yaw_deg)

    def _update_pose_from_mocap(self, frame: MocapFrame) -> None:
        """从 MocapFrame 提取指定 rigid body 的姿态并更新共享变量."""
        target_id = self._config.servo.rigid_body_id
        for rb in frame.rigid_bodies:
            if rb.id == target_id:
                pitch, yaw = _quat_to_pitch_yaw(
                    rb.qx, rb.qy, rb.qz, rb.qw,
                    self._ref_quat,
                )
                # 参照捕获
                if self._ref_quat is None:
                    self._ref_quat = np.array([rb.qw, rb.qx, rb.qy, rb.qz])
                    logger.info("Reference quaternion captured from rigid body %d: "
                                "[%.4f, %.4f, %.4f, %.4f]",
                                target_id, rb.qw, rb.qx, rb.qy, rb.qz)
                with self._pose_lock:
                    self._current_pitch_deg = pitch
                    self._current_yaw_deg = yaw
                break


# ------------------------------------------------------------------
# 四元数工具 (模块级)
# ------------------------------------------------------------------

def _quat_multiply(q1: NDArray, q2: NDArray) -> NDArray:
    """Hamilton 积: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_inverse(q: NDArray) -> NDArray:
    """四元数逆."""
    w, x, y, z = q
    n2 = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z]) / n2


def _quat_to_rotmat(q: NDArray) -> NDArray:
    """四元数 (w,x,y,z) → 3x3 旋转矩阵."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _quat_to_pitch_yaw(
    qx: float, qy: float, qz: float, qw: float,
    ref_quat: NDArray | None,
) -> tuple[float, float]:
    """四元数 → (pitch_deg, yaw_deg)，相对于参考四元数。

    旋转分解约定: R = Ry(yaw) @ Rx(pitch)。
    """
    q = np.array([qw, qx, qy, qz])

    if ref_quat is not None:
        q_rel = _quat_multiply(_quat_inverse(ref_quat), q)
    else:
        q_rel = q

    R = _quat_to_rotmat(q_rel)

    pitch_rad = np.arctan2(-R[1, 2], R[1, 1])
    yaw_rad = np.arctan2(R[0, 2], R[2, 2])

    return (float(np.rad2deg(pitch_rad)), float(np.rad2deg(yaw_rad)))


def _dataclass_dict(obj) -> dict:
    return {
        k: str(v) if isinstance(v, Path) else v
        for k, v in obj.__dict__.items()
    }
