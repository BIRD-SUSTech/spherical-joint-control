"""运行时前馈 + PID 闭环控制（真机）。

u = decode(d_ff + d_fb, p)
d_ff = g(Δq_d, q_meas, gyro, force)      # 数据驱动前馈
d_fb = DiffPID(q_d - q_meas)             # 任务空间 PID 兜底

用法：
    python -m feedback_control.run_feedforward --g-checkpoint <g_checkpoint.pt> --hold 5 -3
    python -m feedback_control.run_feedforward --g-checkpoint <g_checkpoint.pt> --circle 10 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_collection.config import Config
from data_collection.orchestrator import _quat_to_pitch_yaw
from data_collection.sensor_collectors.force_collector import ForceCollector
from data_collection.sensor_collectors.imu_collector import ImuCollector
from data_collection.sensor_collectors.mocap_collector import MocapCollector
from data_collection.servo_controller import SerialServoDriver
from data_collection.utils.data_types import ForceData, ImuPacket, MocapFrame

from .feedforward import DiffPID, FeedforwardController, decode

logger = logging.getLogger(__name__)

LOOP_HZ = 100


def _drain_latest(q: queue.Queue):
    """排空队列返回最新元素（无则 None）。"""
    item = None
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return item


def parse_args():
    p = argparse.ArgumentParser(description="前馈+PID 闭环控制")
    p.add_argument("--g-checkpoint", required=True, help="g_checkpoint.pt 路径")
    p.add_argument("--circle", nargs=2, type=float, metavar=("AMP_DEG", "PERIOD_S"))
    p.add_argument("--hold", nargs=2, type=float, metavar=("PITCH_DEG", "YAW_DEG"))
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--ip", default="10.1.1.198")
    p.add_argument("--port", default="COM5")
    p.add_argument("--rb", type=int, default=0)
    p.add_argument("--imu-address", default="A5:B2:90:FF:4A:12")
    p.add_argument("--force-port", default="COM3")
    p.add_argument("--settle", type=float, default=2.0, help="参考四元数捕获前静置(s)")
    p.add_argument("--kp-pitch", type=float, default=0.04)
    p.add_argument("--kp-yaw", type=float, default=0.04)
    p.add_argument("--dry-run", action="store_true", help="只读动捕、不驱舵机")
    p.add_argument("--out", default="feedback_control/logs")
    return p.parse_args()


def make_reference(args):
    if args.circle:
        amp, period = args.circle

        def ref(t):
            return (amp * math.cos(2 * math.pi * t / period),
                    amp * math.sin(2 * math.pi * t / period))
    elif args.hold:
        a0, a1 = args.hold

        def ref(t):
            return (a0, a1)
    else:

        def ref(t):
            return (0.0, 0.0)
    return ref


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ffc = FeedforwardController(args.g_checkpoint, device=device)
    pid = DiffPID(kp_pitch=args.kp_pitch, kp_yaw=args.kp_yaw)
    ref = make_reference(args)
    p = ffc.pretension
    logger.info("g loaded (device=%s, pretension=%.3f)", device, p)

    cfg = Config()
    cfg.mocap.server_ip = args.ip
    cfg.servo.rigid_body_id = args.rb
    cfg.imu.device_address = args.imu_address
    cfg.force.serial_port = args.force_port

    # 共享状态
    st = {"pitch": 0.0, "yaw": 0.0, "ref_q": None}
    lock = threading.Lock()

    def pose_cb(frame: MocapFrame):
        for rb in frame.rigid_bodies:
            if rb.id == args.rb:
                with lock:
                    if st["ref_q"] is None:
                        st["ref_q"] = np.array([rb.qw, rb.qx, rb.qy, rb.qz])
                    pitch, yaw = _quat_to_pitch_yaw(
                        rb.qx, rb.qy, rb.qz, rb.qw, st["ref_q"])
                    st["pitch"], st["yaw"] = pitch, yaw
                break

    # 采集器
    q_mocap = queue.Queue(maxsize=2000)
    q_imu = queue.Queue(maxsize=2000)
    q_force = queue.Queue(maxsize=2000)
    start_ev, stop_ev = threading.Event(), threading.Event()

    mocap = MocapCollector(cfg.mocap, q_mocap, start_ev, stop_ev, pose_callback=pose_cb)
    imu = ImuCollector(cfg.imu, q_imu, start_ev, stop_ev)
    force = ForceCollector(cfg.force, q_force, start_ev, stop_ev)

    if not mocap.connect():
        logger.error("mocap 连接失败")
        return 1
    mocap.start()
    imu.start()
    force.start()

    # 等参考四元数（球杆保持中立）
    logger.info("保持中立 %ds 以捕获参考四元数...", args.settle)
    deadline = time.time() + args.settle
    while time.time() < deadline:
        time.sleep(0.01)
    with lock:
        if st["ref_q"] is None:
            logger.error("未收到动捕帧")
            return 1
    logger.info("参考四元数已捕获")

    # 舵机
    sink = None
    if not args.dry_run:
        sink = SerialServoDriver(port=args.port)
        if not sink.connect():
            logger.error("舵机串口 %s 打开失败", args.port)
            return 1
        sink.send(np.full(4, p))
        logger.info("舵机已初始化为预紧 %.3f", p)

    # CSV
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"ff_closed_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t_s", "target_pitch", "target_yaw", "current_pitch", "current_yaw",
                "ff_d1", "ff_d2", "fb_d1", "fb_d2",
                "servo_1", "servo_2", "servo_3", "servo_4"])

    logger.info("控制回路启动 (%dHz)", LOOP_HZ)
    t0 = time.perf_counter()
    dt = 1.0 / LOOP_HZ
    next_t = t0

    try:
        while (time.perf_counter() - t0) < args.duration:
            t = time.perf_counter() - t0
            tp, ty = ref(t)
            tnp, tny = ref(t + dt)          # 下一拍目标
            dq_d = np.array([tnp - tp, tny - ty])   # 期望增量

            with lock:
                qp, qy = st["pitch"], st["yaw"]

            # 传感器
            ip = _drain_latest(q_imu)
            gyro = np.array([ip.gyro_x, ip.gyro_y, ip.gyro_z]) if ip else np.zeros(3)
            fp = _drain_latest(q_force)
            force = np.array([fp.ch1, fp.ch2, fp.ch3, fp.ch4]) if fp else np.zeros(4)

            d_ff = ffc.compute(dq_d, np.array([qp, qy]), gyro, force)
            d_fb = pid.update(tp - qp, ty - qy, dt)
            u = decode(d_ff + d_fb, p)

            if sink:
                sink.send(u)

            w.writerow([round(t, 4), round(tp, 4), round(ty, 4),
                        round(qp, 4), round(qy, 4),
                        round(float(d_ff[0]), 5), round(float(d_ff[1]), 5),
                        round(float(d_fb[0]), 5), round(float(d_fb[1]), 5),
                        *[round(float(x), 4) for x in u]])

            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("收到中断")
    finally:
        if sink:
            sink.send(np.zeros(4))
            sink.disconnect()
        stop_ev.set()
        mocap.stop(); mocap.join(timeout=2)
        imu.stop(); imu.join(timeout=2)
        force.stop(); force.join(timeout=2)
        f.close()
        logger.info("已退出，数据: %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
