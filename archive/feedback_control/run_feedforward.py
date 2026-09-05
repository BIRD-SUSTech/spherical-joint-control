"""运行时结构化摩擦前馈 + PID 闭环控制（真机）。

u = decode(d_ff + d_fb, p)
d_ff = FrictionFeedforward(Δq_d)     # 结构化摩擦前馈（只依赖期望增量，无需 IMU/力）
d_fb = DiffPID(q_d - q_meas)         # 任务空间 PID 兜底

用法：
    python -m feedback_control.run_feedforward --friction-config <config> --hold 5 -3
    python -m feedback_control.run_feedforward --friction-config <config> --circle 10 8
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
from data_collection.sensor_collectors.mocap_collector import MocapCollector
from data_collection.servo_controller import SerialServoDriver
from data_collection.utils.data_types import MocapFrame

from .feedforward import DiffPID
from .friction_feedforward import FrictionFeedforward, decode

logger = logging.getLogger(__name__)

LOOP_HZ = 100


def parse_args():
    p = argparse.ArgumentParser(description="结构化摩擦前馈 + PID 闭环")
    p.add_argument("--friction-config", required=True, help="friction_config.json 路径")
    p.add_argument("--circle", nargs=2, type=float, metavar=("AMP_DEG", "PERIOD_S"))
    p.add_argument("--hold", nargs=2, type=float, metavar=("PITCH_DEG", "YAW_DEG"))
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--ip", default="10.1.1.198")
    p.add_argument("--port", default="COM5")
    p.add_argument("--rb", type=int, default=0)
    p.add_argument("--settle", type=float, default=2.0)
    p.add_argument("--kp-pitch", type=float, default=0.04)
    p.add_argument("--kp-yaw", type=float, default=0.04)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-feedforward", action="store_true", help="关闭前馈，仅 PID（对照）")
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

    ffc = FrictionFeedforward(args.friction_config)
    pid = DiffPID(kp_pitch=args.kp_pitch, kp_yaw=args.kp_yaw)
    ref = make_reference(args)
    p = ffc.p
    logger.info("摩擦前馈: G=%s f_c=%s f_s=%s pretension=%.3f",
                np.round(ffc.G, 4), np.round(ffc.f_c, 4), np.round(ffc.f_s, 4), p)

    cfg = Config()
    cfg.mocap.server_ip = args.ip
    cfg.servo.rigid_body_id = args.rb

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

    q_mocap = queue.Queue(maxsize=2000)
    start_ev, stop_ev = threading.Event(), threading.Event()
    mocap = MocapCollector(cfg.mocap, q_mocap, start_ev, stop_ev, pose_callback=pose_cb)

    if not mocap.connect():
        logger.error("mocap 连接失败")
        return 1
    mocap.start()

    logger.info("保持中立 %ds 以捕获参考四元数...", args.settle)
    deadline = time.time() + args.settle
    while time.time() < deadline:
        time.sleep(0.01)
    with lock:
        if st["ref_q"] is None:
            logger.error("未收到动捕帧")
            return 1
    logger.info("参考四元数已捕获")

    sink = None
    if not args.dry_run:
        sink = SerialServoDriver(port=args.port)
        if not sink.connect():
            logger.error("舵机串口 %s 打开失败", args.port)
            return 1
        sink.send(np.full(4, p))
        logger.info("舵机已初始化为预紧 %.3f", p)

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
            tnp, tny = ref(t + dt)
            dq_d = np.array([tnp - tp, tny - ty])

            with lock:
                qp, qy = st["pitch"], st["yaw"]

            d_ff = np.zeros(2) if args.no_feedforward else ffc.compute(dq_d)
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
        mocap.stop()
        mocap.join(timeout=2)
        f.close()
        logger.info("已退出，数据: %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
