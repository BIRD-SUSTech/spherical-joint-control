"""闭环采集编排：多线程 → queue → 统一 schema CSV（M2，L1 采集层）。

组合 L0（hardware.mocap / hardware.servo）+ L1（imu/force 采集器）+ L3（control.pid），
把闭环控制与多传感器数据统一落盘到一套 schema（含 segment_id / phase 语义）。

用法：
    python -m collect.orchestrator --mock --duration 2          # 离线自检
    python -m collect.orchestrator --hold 0 0 --duration 30     # 闭环持中位 + 采集
    python -m collect.orchestrator --circle 10 6 --duration 30  # 圆形轨迹 + 采集
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from collect.schema import (
    CollectionPhase,
    FORCE_CSV_COLUMNS,
    IMU_CSV_COLUMNS,
    MOCAP_CSV_COLUMNS,
    SERVO_CSV_COLUMNS,
    ForceData,
    ImuPacket,
    MocapFrame,
    RigidBody,
    ServoState,
)
from collect.sensors.force import ForceCollector
from collect.sensors.imu import ImuCollector
from collect.session import SessionManager
from collect.writer import CsvWriter
from control.calibration import Calibration
from control.controller_config import ControllerConfig
from control.trajectory import make_traj
from control.pid import PIDController
from excite.guardian import Guardian
from excite.signals import default_segments, extended_segments, sample_segment
from hardware.mocap import MockMocap, MocapReader, Pose
from hardware.servo import MockServoBus, ServoBus

logger = logging.getLogger(__name__)

LOOP_HZ = 100
KP = 8.0
KI = 1.0
KD = 2.5
LIMIT = 400.0
DEADBAND = 0.2
ALPHA = 0.3


QUEUE_MAXSIZE = 5000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="球关节闭环采集")
    p.add_argument("--mock", action="store_true", help="离线自检：合成动捕，不接任何硬件")
    p.add_argument("--dry-run", action="store_true", help="只读动捕、不驱动舵机")
    p.add_argument("--hold", nargs=2, type=float, metavar=("FRONT_BACK_DEG", "LEFT_RIGHT_DEG"),
                   help="恒定目标（前后°, 左右°）")
    p.add_argument("--circle", nargs=2, type=float, metavar=("AMP_DEG", "PERIOD_S"),
                   help="圆形轨迹（振幅°, 周期 s）")
    p.add_argument("--lissajous", nargs=4, type=float,
                   metavar=("AMP_FB", "AMP_LR", "F1", "F2"), help="Lissajous 双频 2D")
    p.add_argument("--eight", nargs=2, type=float, metavar=("AMP", "PERIOD_S"),
                   help="8 字形（1:2 频率比，换向频繁）")
    p.add_argument("--variable-circle", nargs=2, type=float, metavar=("AMP", "PERIOD_S"),
                   help="变速圆（快慢交替）")
    p.add_argument("--waypoints", nargs="+", type=float, metavar="FB LR DUR",
                   help="点到点（fb lr dur 循环，多组）")
    p.add_argument("--duration", type=float, default=30.0, help="运行时长 s（默认 30）")
    p.add_argument("--ip", default="10.1.1.198", help="动捕服务器 IP")
    p.add_argument("--servo-port", default=None, help="舵机串口（真实模式必填）")
    p.add_argument("--calibration", default=None, help="标定 JSON（缺省用默认映射）")
    p.add_argument("--controller-config", default=None,
                   help="控制器参数 JSON（前馈增益等；缺省=无前馈 u_ff=0）")
    p.add_argument("--ab", action="store_true", help="自动 A/B：同轨迹跑 baseline + 前馈两段")
    p.add_argument("--baseline-controller-config", default=None,
                   help="A/B baseline 段的控制器参数 JSON（缺省=u_ff=0；可传上一轮优化参数）")
    p.add_argument("--inter-segment-settle", type=float, default=5.0,
                   help="A/B 段间回正时长 s（默认 5，持续发 0 让球杆回到中立平衡态）")
    p.add_argument("--rb", type=int, default=0, help="动捕刚体索引")
    p.add_argument("--no-imu", action="store_true", help="不采集 IMU")
    p.add_argument("--no-force", action="store_true", help="不采集力传感器")
    p.add_argument("--force-port", default=None, help="力传感器串口（启用采集时必填）")
    p.add_argument("--out", default="collect/logs", help="会话根目录")
    p.add_argument("--segment-id", type=int, default=0, help="闭环数据段 id（默认 0）")
    p.add_argument("--open-loop", action="store_true", help="开环激励模式（替代闭环）")
    p.add_argument("--open-loop-fs", type=float, default=100.0, help="开环激励频率 Hz")
    p.add_argument("--open-loop-duration", type=float, default=None,
                   help="开环总时长上限 s（缺省=跑完所有激励段）")
    p.add_argument("--extended", action="store_true", help="用补数据扩展激励段（M5 充分采集）")
    return p.parse_args()





class Orchestrator:
    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._traj = make_traj(args)
        self._segment_id = args.segment_id
        self._phase = CollectionPhase.EXPLORATION
        self._calib = Calibration.load(args.calibration) if args.calibration else Calibration.default()
        self._controller = (ControllerConfig.load(args.controller_config)
                            if args.controller_config else ControllerConfig.none())
        self._baseline_controller = (ControllerConfig.load(args.baseline_controller_config)
                                     if args.baseline_controller_config else None)

        self._stop_event = threading.Event()
        self._q_mocap = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._q_imu = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._q_force = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._q_servo = queue.Queue(maxsize=QUEUE_MAXSIZE)

        self._mocap = None
        self._imu = None
        self._force = None
        self._bus = None
        self._writers: dict[str, CsvWriter] = {}
        self._session = None
        self._dropped = {"mocap": 0, "servo": 0}
        self._guardian = None

    # ------------------------------------------------------------------

    def run(self) -> int:
        args = self._args
        consumers: list[threading.Thread] = []
        rc = 1
        try:
            # 1. 会话目录 + CSV 写入器
            self._session = SessionManager(root_dir=Path(args.out)).create_session()
            logger.info("会话目录: %s", self._session.dir)
            self._open_writers()

            # 2. 动捕（on_frame 回调 → mocap 落盘）
            mocap_cb = self._on_mocap_frame if not args.mock else None
            self._mocap = MockMocap(on_frame=self._on_mocap_frame) if args.mock \
                else MocapReader(args.ip, args.rb, on_frame=mocap_cb)
            if not self._mocap.connect():
                logger.error("动捕连接失败，退出")
                return 1
            self._mocap.start()

            # 真实动捕等待首帧
            if not args.mock:
                deadline = time.time() + 3.0
                while time.time() < deadline and self._mocap.get_pose() is None:
                    time.sleep(0.01)
                if self._mocap.get_pose() is None:
                    logger.error("3s 内未收到动捕帧，退出")
                    return 1

            # 3. 舵机总线
            if args.mock or args.dry_run:
                self._bus = MockServoBus()
            elif args.servo_port is None:
                logger.error("真实模式必须指定 --servo-port（舵机串口）")
                return 1
            else:
                self._bus = ServoBus(args.servo_port)
            if not self._bus.connect():
                return 1

            # 4. IMU / 力采集器（启动失败降级，不中断主流程）
            if not args.mock and not args.no_imu:
                try:
                    self._imu = ImuCollector(self._q_imu, self._stop_event)
                    self._imu.start()
                except Exception:
                    logger.exception("IMU 启动失败，跳过 IMU 采集")
                    self._imu = None
            if not args.mock and not args.no_force:
                if args.force_port is None:
                    logger.warning("已启用力采集但未指定 --force-port，跳过力采集")
                else:
                    try:
                        self._force = ForceCollector(self._q_force, self._stop_event,
                                                     serial_port=args.force_port)
                        self._force.start()
                    except Exception:
                        logger.exception("力传感器启动失败，跳过力采集")
                        self._force = None

            # 5. consumer 线程
            consumers = self._start_consumers()

            # 6. guardian（开环/闭环都兜底，急停=回中位 70°）
            self._guardian = Guardian(self._mocap, self._bus, self._calib)
            self._guardian.start()

            # 7. 控制循环（开环激励 或 闭环 PID）
            if args.open_loop:
                segs = extended_segments() if args.extended else default_segments()
                logger.info("开环激励启动（%dHz），段数=%d（%s）",
                            args.open_loop_fs, len(segs),
                            "extended" if args.extended else "default")
                self._run_open_loop(args.open_loop_fs, args.open_loop_duration)
            elif args.ab:
                base_ctrl = self._baseline_controller
                logger.info("A/B 对照：段0 baseline(%s) + 段1 前馈",
                            "u_ff=0" if base_ctrl is None else "上一轮优化参数")
                self._run_control_loop(args.duration, segment_id=0,
                                       controller=base_ctrl or ControllerConfig.none())
                self._settle_neutral(args.inter_segment_settle)  # 段间回正
                self._run_control_loop(args.duration, segment_id=1,
                                       controller=self._controller)
            else:
                mode = "mock" if args.mock else ("dry-run" if args.dry_run else f"串口 {args.servo_port}")
                logger.info("闭环采集启动（%dHz），模式=%s，时长 %.0fs",
                            LOOP_HZ, mode, args.duration)
                self._run_control_loop(args.duration)

            # 8. 收尾：回中位
            logger.info("收尾：回中位 (offset=0, offset=0)")
            self._bus.send_pair(0, 0)
            time.sleep(1.5)
            rc = 0
        except Exception:
            logger.exception("采集异常终止")
            rc = 1
        finally:
            self._finalize(consumers)
        return rc

    def _finalize(self, consumers: list[threading.Thread]) -> None:
        """统一收尾（正常/异常路径都走）：回中位兜底 → 停采集 → 排空 → 关 CSV → 尽力写 metadata。"""
        if self._bus is not None:
            try:
                self._bus.send_pair(0, 0)
            except Exception:  # noqa: BLE001
                pass

        self._stop_event.set()
        if self._guardian is not None:
            try:
                self._guardian.stop()
                self._guardian.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
        for c in (self._mocap, self._imu, self._force):
            if c is not None:
                try:
                    c.stop()
                    c.join(timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass

        for t in consumers:
            try:
                t.join()
            except Exception:  # noqa: BLE001
                pass

        self._close_writers()

        if self._session is not None:
            try:
                self._write_metadata()
            except Exception:  # noqa: BLE001
                logger.exception("metadata 写入失败")

    # ------------------------------------------------------------------
    # 动捕回调 / 控制循环
    # ------------------------------------------------------------------

    def _on_mocap_frame(self, pose: Pose) -> None:
        now_ns = time.perf_counter_ns()
        now_ms = int(time.time() * 1000)
        rb = RigidBody(
            id=self._args.rb,
            x=pose.x, y=pose.y, z=pose.z,
            qx=pose.qx, qy=pose.qy, qz=pose.qz, qw=pose.qw,
        )
        frame = MocapFrame(
            pc_timestamp_ns=now_ns,
            pc_receive_unix_time_ms=now_ms,
            frame_index=pose.frame_index,
            hardware_timestamp=0,
            rigid_bodies=[rb],
        )
        try:
            self._q_mocap.put_nowait(frame)
        except queue.Full:
            self._dropped["mocap"] += 1

    def _settle_neutral(self, settle_s: float) -> None:
        """回中位并保持 settle_s 秒（持续发 0），让球杆稳定到 offset=0 平衡态。

        A/B 两段从相同的中立状态出发：段间持续发 offset=0（保持张紧中立位），
        球杆回到该指令对应的平衡态，两段起点一致。
        """
        logger.info("段间回正 %.1fs（持续发 offset=0）", settle_s)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < settle_s:
            self._bus.send_pair(0, 0)
            time.sleep(0.05)

    def _run_control_loop(self, duration_s: float, segment_id: int | None = None,
                          controller=None) -> None:
        pid_fb = PIDController(kp=KP, ki=KI, kd=KD, limit=LIMIT,
                               deadband=DEADBAND, alpha=ALPHA)
        pid_lr = PIDController(kp=KP, ki=KI, kd=KD, limit=LIMIT,
                               deadband=DEADBAND, alpha=ALPHA)

        t0 = time.perf_counter()
        dt = 1.0 / LOOP_HZ
        next_t = t0
        prev_t_fb = prev_t_lr = None

        while not self._stop_event.is_set():
            if self._guardian is not None and self._guardian.is_triggered:
                return
            t = time.perf_counter() - t0
            if t >= duration_s:
                break
            t_fb, t_lr = self._traj(t)
            qdot_fb = (t_fb - prev_t_fb) / dt if prev_t_fb is not None else 0.0
            qdot_lr = (t_lr - prev_t_lr) / dt if prev_t_lr is not None else 0.0
            prev_t_fb, prev_t_lr = t_fb, t_lr

            pose = self._mocap.get_pose()
            if pose is None:
                time.sleep(dt)
                continue

            curr_fb, curr_lr = self._calib.map_pose(pose.roll, pose.pitch)

            pid_fb.target = t_fb
            pid_lr.target = t_lr
            out_fb = int(pid_fb.calculate(curr_fb))
            out_lr = int(pid_lr.calculate(curr_lr))
            ctrl = controller if controller is not None else self._controller
            if ctrl.has_feedforward():
                ff_fb, ff_lr = ctrl.feedforward(t_fb, t_lr, qdot_fb, qdot_lr)
                out_fb = int(ff_fb + out_fb)
                out_lr = int(ff_lr + out_lr)
            self._bus.send_pair(out_fb, out_lr)

            state = ServoState(
                pc_timestamp_ns=time.perf_counter_ns(),
                pc_receive_unix_time_ms=int(time.time() * 1000),
                t_s=round(t, 4),
                target_front_back_deg=t_fb,
                target_left_right_deg=t_lr,
                current_front_back_deg=curr_fb,
                current_left_right_deg=curr_lr,
                servo_front_back_offset=out_fb,
                servo_left_right_offset=out_lr,
                segment_id=segment_id if segment_id is not None else self._segment_id,
            )
            try:
                self._q_servo.put_nowait(state)
            except queue.Full:
                self._dropped["servo"] += 1

            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()  # 掉拍重新对齐

    def _run_open_loop(self, fs: float, duration_s: float | None = None) -> None:
        """开环激励执行：逐段发 offset，每段一个 segment_id，guardian 触发即停。"""
        dt = 1.0 / fs
        t_start = time.perf_counter()
        segments = extended_segments() if self._args.extended else default_segments()
        gains = self._calib.gain_deg_per_offset
        for seg_id, seg in enumerate(segments):
            if duration_s is not None and time.perf_counter() - t_start >= duration_s:
                break
            t_seq, fb_seq, lr_seq = sample_segment(seg, fs,
                                                   gains["front_back"], gains["left_right"])
            logger.info("激励段 %d/%d: kind=%s 采样数=%d",
                        seg_id, len(segments), seg["kind"], len(t_seq))
            for i in range(len(t_seq)):
                if self._stop_event.is_set() or self._guardian.is_triggered:
                    return
                if duration_s is not None and time.perf_counter() - t_start >= duration_s:
                    return
                fb_off = int(fb_seq[i])
                lr_off = int(lr_seq[i])
                self._bus.send_pair(fb_off, lr_off)

                pose = self._mocap.get_pose()
                curr_fb = curr_lr = 0.0
                if pose is not None:
                    curr_fb, curr_lr = self._calib.map_pose(pose.roll, pose.pitch)

                state = ServoState(
                    pc_timestamp_ns=time.perf_counter_ns(),
                    pc_receive_unix_time_ms=int(time.time() * 1000),
                    t_s=round(t_seq[i], 4),
                    target_front_back_deg=0.0,   # 开环无目标角
                    target_left_right_deg=0.0,
                    current_front_back_deg=curr_fb,
                    current_left_right_deg=curr_lr,
                    servo_front_back_offset=fb_off,
                    servo_left_right_offset=lr_off,
                    segment_id=seg_id,
                )
                try:
                    self._q_servo.put_nowait(state)
                except queue.Full:
                    self._dropped["servo"] += 1

                next_t = time.perf_counter() + dt
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        logger.info("开环激励完成")

    # ------------------------------------------------------------------
    # CSV 写入器 + consumer
    # ------------------------------------------------------------------

    def _open_writers(self) -> None:
        sp = self._session
        self._writers = {
            "mocap": CsvWriter(sp.mocap_csv, MOCAP_CSV_COLUMNS),
            "servo": CsvWriter(sp.servo_csv, SERVO_CSV_COLUMNS),
        }
        if self._imu_enabled():
            self._writers["imu"] = CsvWriter(sp.imu_csv, IMU_CSV_COLUMNS)
        if self._force_enabled():
            self._writers["force"] = CsvWriter(sp.force_csv, FORCE_CSV_COLUMNS)
        for w in self._writers.values():
            w.open()

    def _imu_enabled(self) -> bool:
        return not self._args.mock and not self._args.no_imu

    def _force_enabled(self) -> bool:
        return not self._args.mock and not self._args.no_force

    def _start_consumers(self) -> list[threading.Thread]:
        threads = []
        for name, q in [
            ("mocap", self._q_mocap),
            ("imu", self._q_imu),
            ("force", self._q_force),
            ("servo", self._q_servo),
        ]:
            writer = self._writers.get(name)
            if writer is None:
                continue
            t = threading.Thread(target=self._consumer, args=(name, q, writer),
                                 name=f"Consumer-{name}", daemon=True)
            t.start()
            threads.append(t)
        return threads

    def _consumer(self, name: str, q: queue.Queue, writer: CsvWriter) -> None:
        while not self._stop_event.is_set():
            try:
                item = q.get(timeout=0.1)
                self._write_item(writer, item)
            except queue.Empty:
                continue
        while True:  # 排空残余
            try:
                item = q.get_nowait()
                self._write_item(writer, item)
            except queue.Empty:
                break
        logger.info("Consumer-%s 停止，%d 行", name, writer.row_count)

    def _write_item(self, writer: CsvWriter, item) -> None:
        phase = self._phase.value
        if isinstance(item, MocapFrame):
            for row in item.to_csv_rows():
                row["phase"] = phase
                writer.write_row(row)
        elif isinstance(item, ImuPacket):
            row = item.to_csv_row()
            row["phase"] = phase
            writer.write_row(row)
        elif isinstance(item, ForceData):
            row = item.to_csv_row()
            row["phase"] = phase
            writer.write_row(row)
        elif isinstance(item, ServoState):
            row = item.to_csv_row()
            row["phase"] = phase
            writer.write_row(row)

    # ------------------------------------------------------------------
    # 停止 / 清理 / 元数据
    # ------------------------------------------------------------------

    def _close_writers(self) -> None:
        for w in self._writers.values():
            w.close()

    def _write_metadata(self) -> None:
        row_counts = {name: w.row_count for name, w in self._writers.items()}
        dropped = dict(self._dropped)
        if self._imu:
            dropped["imu"] = self._imu.dropped_count
        if self._force:
            dropped["force"] = self._force.dropped_count
        metadata = {
            "session_dir": str(self._session.dir),
            "started_at": datetime.now().isoformat(),
            "mode": "mock" if self._args.mock else ("dry-run" if self._args.dry_run else ("open_loop" if self._args.open_loop else "closed_loop")),
            "segment_id": self._segment_id,
            "row_counts": row_counts,
            "dropped_frames": dropped,
            "config": {
                "trajectory": _trajectory_info(self._args),
                "calibration": self._args.calibration,
                "controller_config": self._args.controller_config,
                "baseline_controller_config": self._args.baseline_controller_config,
                "controller": {
                    "gain_poly": self._controller.gain_poly,
                    "direction_gains": self._controller.direction_gains,
                    "slew_limit": self._controller.slew_limit,
                    "hysteresis": self._controller.hysteresis,
                },
                "servo_port": self._args.servo_port,
                "force_port": self._args.force_port,
                "duration": self._args.duration,
                "ab": self._args.ab,
                "inter_segment_settle": self._args.inter_segment_settle,
            },
        }
        self._session.metadata_json.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


def _trajectory_info(args) -> dict:
    """从 args 提取轨迹类型 + 参数。"""
    for name in ("circle", "hold", "lissajous", "eight", "variable_circle", "waypoints"):
        val = getattr(args, name, None)
        if val is not None:
            return {"type": name, "params": val}
    return {"type": "idle", "params": None}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    return Orchestrator(args).run()


if __name__ == "__main__":
    sys.exit(main())
