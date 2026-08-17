"""闭环反馈控制主程序（最小实现，从 CloseLoop 参考代码重写）。

控制结构（与参考一致）：
    mocap 欧拉角 (pitch=左右, yaw=前后)
        → 两个独立 PID（无 IK、无缆长模型）
        → 2 DOF 控制量 → 4 舵机角度（主机侧对侧耦合）
        → 当前固件协议（逗号分隔 4 个角度）

与参考代码的 I/O 差异（已转换）：
    输入：参考直接读 SDK 扩展欧拉角；本实现优先用 SDK 欧拉角，缺失时退化为四元数分解。
    输出：参考发 2 路二进制 0xAA id int16 0x55（固件内耦合）；本实现发 4 路角度（固件不耦合，主机耦合）。

命名沿用当前系统 pitch / yaw：
    pitch = DOF1 = 舵机对 1↔3；yaw = DOF2 = 舵机对 2↔4。

只依赖动捕与串口，IMU/力传感器已完全解耦（不导入）。

用法：
    python -m feedback_control.main --dry-run          # 只读动捕、不驱动舵机（首测推荐）
    python -m feedback_control.main                    # 保持中立位
    python -m feedback_control.main --circle 10 6      # 圆形轨迹 10°、6s
    python -m feedback_control.main --hold 5 -3        # 恒定目标 (pitch=5°, yaw=-3°)
    python -m feedback_control.main --mock --duration 5  # 无硬件离线自检
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

from .mocap import MockMocap, MocapReader
from .pid import PIDController
from .servo import coupled_angles, format_command

logger = logging.getLogger(__name__)

# ---- 可调参数（起始值，需实机整定）----
KP = 3.0
KI = 0.3
KD = 0.8
DEADBAND = 0.2
ALPHA = 0.3
LOOP_HZ = 100
FLIP_PITCH = False   # 若 pitch 方向反了改 True
FLIP_YAW = False     # 若 yaw 方向反了改 True


class SerialSink:
    def __init__(self, port: str):
        self.port = port
        self._ser = None

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(self.port, 115200, timeout=0.1)
            return True
        except serial.SerialException as e:
            logger.error("串口 %s 打开失败: %s", self.port, e)
            return False

    def send(self, data: bytes) -> None:
        if self._ser and self._ser.is_open:
            self._ser.write(data)

    def close(self) -> None:
        if self._ser:
            self._ser.close()
            self._ser = None


class LogSink:
    """不接串口时打印指令（--mock/--dry-run 用）。"""

    def connect(self) -> bool:
        return True

    def send(self, data: bytes) -> None:
        logger.info("cmd: %s", data.decode().strip())

    def close(self) -> None:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="球关节闭环反馈控制")
    p.add_argument("--mock", action="store_true", help="离线自检：合成动捕，不接任何硬件")
    p.add_argument("--dry-run", action="store_true", help="只读动捕、不驱动舵机")
    p.add_argument("--circle", nargs=2, type=float, metavar=("AMP_DEG", "PERIOD_S"),
                   help="圆形轨迹（振幅°, 周期 s）")
    p.add_argument("--hold", nargs=2, type=float, metavar=("PITCH_DEG", "YAW_DEG"),
                   help="恒定目标（pitch°, yaw°）")
    p.add_argument("--duration", type=float, default=30.0, help="运行时长 s（默认 30）")
    p.add_argument("--ip", default="10.1.1.198", help="mocap 服务器 IP")
    p.add_argument("--port", default="COM5", help="舵机串口")
    p.add_argument("--rb", type=int, default=0, help="mocap 刚体索引")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # 轨迹
    if args.circle:
        amp, period = args.circle
        def traj(t):
            return (amp * math.sin(2 * math.pi * t / period),
                    amp * math.cos(2 * math.pi * t / period))
    elif args.hold:
        a0, a1 = args.hold
        def traj(t):
            return (a0, a1)
    else:
        def traj(t):
            return (0.0, 0.0)

    # 动捕（仅动捕）
    mocap = MockMocap() if args.mock else MocapReader(args.ip, args.rb)
    if not mocap.connect():
        logger.error("mocap 连接失败，退出")
        return 1
    mocap.start()

    # 等待首帧（真实动捕必须要有反馈，否则舵机会被推满）
    if not args.mock:
        deadline = time.time() + 3.0
        while time.time() < deadline and mocap.get_pose().frame_index < 0:
            time.sleep(0.01)
        if mocap.get_pose().frame_index < 0:
            logger.error("3s 内未收到动捕帧，检查服务器/刚体索引，退出")
            return 1

    # 串口
    sink = LogSink() if (args.mock or args.dry_run) else SerialSink(args.port)
    if not sink.connect():
        return 1

    pid_pitch = PIDController(kp=KP, ki=KI, kd=KD, deadband=DEADBAND, alpha=ALPHA)
    pid_yaw = PIDController(kp=KP, ki=KI, kd=KD, deadband=DEADBAND, alpha=ALPHA)

    # CSV
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    csv_path = log_dir / f"closed_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["t_s", "target_pitch", "target_yaw",
                     "pitch", "yaw", "out_pitch", "out_yaw"])

    mode = "mock" if args.mock else ("dry-run" if args.dry_run else f"串口 {args.port}")
    logger.info("控制回路启动（%dHz），模式=%s", LOOP_HZ, mode)

    t0 = time.perf_counter()
    dt = 1.0 / LOOP_HZ
    next_t = t0

    try:
        while (time.perf_counter() - t0) < args.duration:
            t = time.perf_counter() - t0
            t_pitch, t_yaw = traj(t)

            pose = mocap.get_pose()
            pid_pitch.target = t_pitch
            pid_yaw.target = t_yaw
            out_pitch = pid_pitch.calculate(pose.pitch)
            out_yaw = pid_yaw.calculate(pose.yaw)

            angles = coupled_angles(out_pitch, out_yaw, FLIP_PITCH, FLIP_YAW)
            sink.send(format_command(angles))

            writer.writerow([round(t, 4), round(t_pitch, 4), round(t_yaw, 4),
                             round(pose.pitch, 4), round(pose.yaw, 4),
                             round(out_pitch, 4), round(out_yaw, 4)])

            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()  # 掉拍则重新对齐
    except KeyboardInterrupt:
        logger.info("收到中断，停止")
    finally:
        f.close()
        sink.close()
        mocap.stop()
        logger.info("已退出，数据: %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
