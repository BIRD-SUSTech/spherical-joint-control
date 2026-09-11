"""闭环主循环：最小闭环（双独立 PID，offset 空间）。

行为级复刻 archive/example_code/steering_motor/Scripts/CloseLoop/main.py：
    mocap 欧拉角 → 双 PID → 2 路差分 offset → 固件内耦合的 2 组舵机。

物理命名（设计文档 §6）：
    front_back（前后）← 动捕 roll（id=1，默认映射，M3 标定确认）
    left_right（左右）← 动捕 pitch（id=2，默认映射，M3 标定确认）

用法：
    python -m control.loop --mock --duration 2      # 离线自检（无硬件）
    python -m control.loop --dry-run                # 只读动捕、不驱动舵机
    python -m control.loop --hold 0 0               # 保持中立位（前后=0, 左右=0）
    python -m control.loop --hold 5 -3              # 恒定目标
    python -m control.loop --circle 10 6            # 圆形轨迹 10°、6s
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from control.calibration import Calibration
from control.controller_config import ControllerConfig
from control.pid import PIDController
from control.trajectory import make_traj
from hardware.mocap import MockMocap, MocapReader
from hardware.servo import MockServoBus, ServoBus

logger = logging.getLogger(__name__)

LOOP_HZ = 100
KP = 8.0
KI = 1.0
KD = 2.5
LIMIT = 600.0
DEADBAND = 0.2
ALPHA = 0.3



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="球关节最小闭环控制")
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
    p.add_argument("--startup-fade", type=float, default=1.0,
                   help="前馈启动渐入时长 s（0=不渐入；启动后该秒内按比例放行前馈）")
    p.add_argument("--kp", type=float, default=8.0, help="PID 比例增益")
    p.add_argument("--ki", type=float, default=1.0, help="PID 积分增益")
    p.add_argument("--kd", type=float, default=2.5, help="PID 微分增益")
    p.add_argument("--deadband", type=float, default=0.2, help="PID 死区（度）")
    p.add_argument("--alpha", type=float, default=0.3, help="反馈低通系数")
    p.add_argument("--rb", type=int, default=0, help="动捕刚体索引")
    p.add_argument("--no-csv", action="store_true", help="不写闭环 CSV")
    return p.parse_args()





def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    traj = make_traj(args)
    calib = Calibration.load(args.calibration) if args.calibration else Calibration.default()
    controller = (ControllerConfig.load(args.controller_config)
                  if args.controller_config else ControllerConfig.none())

    # 动捕（仅动捕）
    mocap = MockMocap() if args.mock else MocapReader(args.ip, args.rb)
    if not mocap.connect():
        logger.error("动捕连接失败，退出")
        return 1
    mocap.start()

    # 真实动捕必须等到首帧，否则舵机会被推满
    if not args.mock:
        deadline = time.time() + 3.0
        while time.time() < deadline and mocap.get_pose() is None:
            time.sleep(0.01)
        if mocap.get_pose() is None:
            logger.error("3s 内未收到动捕帧，检查服务器/刚体索引，退出")
            return 1

    # 舵机总线
    if args.mock or args.dry_run:
        bus = MockServoBus()
    elif args.servo_port is None:
        logger.error("真实模式必须指定 --servo-port（舵机串口）")
        return 1
    else:
        bus = ServoBus(args.servo_port)
    if not bus.connect():
        return 1

    pid_fb = PIDController(kp=args.kp, ki=args.ki, kd=args.kd, limit=LIMIT,
                           deadband=args.deadband, alpha=args.alpha)
    pid_lr = PIDController(kp=args.kp, ki=args.ki, kd=args.kd, limit=LIMIT,
                           deadband=args.deadband, alpha=args.alpha)

    # CSV（M2 起由 collect 层统一 schema；此处为最小列，物理命名）
    csv_path = None
    f = None
    writer = None
    if not args.no_csv:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(exist_ok=True)
        csv_path = log_dir / f"closed_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        f = open(csv_path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["t_s",
                         "target_front_back", "target_left_right",
                         "current_front_back", "current_left_right",
                         "out_front_back", "out_left_right"])

    mode = "mock" if args.mock else ("dry-run" if args.dry_run else f"串口 {args.servo_port}")
    logger.info("控制回路启动（%dHz），模式=%s", LOOP_HZ, mode)

    t0 = time.perf_counter()
    dt = 1.0 / LOOP_HZ
    next_t = t0
    prev_t_fb = prev_t_lr = None
    prev_qdot_fb = prev_qdot_lr = 0.0

    try:
        while (time.perf_counter() - t0) < args.duration:
            t = time.perf_counter() - t0
            t_fb, t_lr = traj(t)
            qdot_fb = (t_fb - prev_t_fb) / dt if prev_t_fb is not None else 0.0
            qdot_lr = (t_lr - prev_t_lr) / dt if prev_t_lr is not None else 0.0
            qddot_fb = (qdot_fb - prev_qdot_fb) / dt
            qddot_lr = (qdot_lr - prev_qdot_lr) / dt
            prev_t_fb, prev_t_lr = t_fb, t_lr
            prev_qdot_fb, prev_qdot_lr = qdot_fb, qdot_lr

            pose = mocap.get_pose()
            if pose is None:
                time.sleep(dt)
                continue

            curr_fb, curr_lr = calib.map_pose(pose.roll, pose.pitch)

            pid_fb.target = t_fb
            pid_lr.target = t_lr
            out_fb = pid_fb.calculate(curr_fb)
            out_lr = pid_lr.calculate(curr_lr)

            if controller.has_feedforward():
                ff_fb, ff_lr = controller.feedforward(t_fb, t_lr, qdot_fb, qdot_lr,
                                                      qddot_fb, qddot_lr)
                fade = min(1.0, t / args.startup_fade) if args.startup_fade > 0 else 1.0
                out_fb = fade * ff_fb + out_fb
                out_lr = fade * ff_lr + out_lr

            c_fb, c_lr = calib.common_mode(curr_fb, curr_lr)
            bus.send_pair_tension(int(out_fb), int(out_lr), c_fb, c_lr)

            if writer is not None:
                writer.writerow([round(t, 4), round(t_fb, 4), round(t_lr, 4),
                                 round(curr_fb, 4), round(curr_lr, 4),
                                 int(out_fb), int(out_lr)])

            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()  # 掉拍则重新对齐

    except KeyboardInterrupt:
        logger.info("收到中断，停止")
    finally:
        if f is not None:
            f.close()
        # 正常收尾回中位：offset=0，两对舵机回到固件 reset 中立位(Sx_MID)，保持缆张紧。
        # 不用 bus.relax()(id=0)：固件会四路完全放线，导致过度放线/杆垂落（实机反馈）。
        # id=0 松缆仅留给急停/guardian(关节角超限)场景。
        logger.info("收尾：回中位 (offset=0, offset=0)")
        bus.send_pair_tension(0, 0, 0, 0)  # 差分 + 共模同时回零
        time.sleep(1.5)  # 等待舵机回到中位
        bus.close()
        mocap.stop()
        if csv_path is not None:
            logger.info("已退出，数据: %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
