"""符号/增益标定：单轴微幅阶跃 → 2×2 增益矩阵 + 符号（M3）。

数据驱动、不猜测：直接发单轴 offset 阶跃，用动捕实测响应拟合增益与符号，
替代 FLIP 硬编码。输出 JSON 配置供闭环加载（--calibration）。

用法：
    python -m control.calibrate --mock                            # 离线自检（流程跑通）
    python -m control.calibrate --servo-port COM5 --out calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from hardware.mocap import MockMocap, MocapReader
from hardware.servo import MockServoBus, ServoBus

logger = logging.getLogger(__name__)

STEP_OFFSETS = [20, 40, 60]   # 单轴阶跃幅度（offset，微幅覆盖死区以上）
HOLD_S = 3.0                  # 每步保持（等稳态）
SETTLE_S = 1.0                # 回中位后静置
MIN_DEADBAND_DEG = 0.05       # 死区判定阈值（度）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="符号/增益标定（单轴微幅阶跃）")
    p.add_argument("--mock", action="store_true", help="离线自检：合成动捕，不接硬件")
    p.add_argument("--servo-port", default=None, help="舵机串口（真实模式必填）")
    p.add_argument("--ip", default="10.1.1.198", help="动捕服务器 IP")
    p.add_argument("--rb", type=int, default=0, help="动捕刚体索引")
    p.add_argument("--out", default="calibration.json", help="标定 JSON 输出路径")
    p.add_argument("--csv", default=None, help="可选：标定样本 CSV 落盘路径")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 拟合（纯函数，离线可测）
# ---------------------------------------------------------------------------

def _slope(pairs) -> float:
    """过原点最小二乘斜率 g = Σ(x·y)/Σ(x²)。pairs: list[(x, y)]。"""
    if not pairs:
        return 0.0
    xx = sum(float(p[0]) * float(p[0]) for p in pairs)
    if xx == 0.0:
        return 0.0
    xy = sum(float(p[0]) * float(p[1]) for p in pairs)
    return xy / xx


def fit_gain_matrix(samples) -> list[list[float]]:
    """samples: list[(axis_id, offset, droll, dpitch)] → 2×2 增益矩阵。

    Δ[roll, pitch] = G @ Δ[offset_1, offset_2]
    G = [[g_roll_1, g_roll_2],
         [g_pitch_1, g_pitch_2]]
    """
    g_roll_1 = _slope([(off, droll) for aid, off, droll, _ in samples if aid == 1])
    g_pitch_1 = _slope([(off, dpitch) for aid, off, _, dpitch in samples if aid == 1])
    g_roll_2 = _slope([(off, droll) for aid, off, droll, _ in samples if aid == 2])
    g_pitch_2 = _slope([(off, dpitch) for aid, off, _, dpitch in samples if aid == 2])
    return [[g_roll_1, g_roll_2], [g_pitch_1, g_pitch_2]]


def derive_calibration(G: list[list[float]]) -> dict:
    """从 2×2 增益矩阵推导标定配置（前后/左右 对应哪个欧拉角 + 符号 + 增益）。"""
    g_roll_1, g_roll_2 = G[0]
    g_pitch_1, g_pitch_2 = G[1]

    # 前后(id=1)主要驱动哪个欧拉角（取增益绝对值大者）
    fb_euler = "roll" if abs(g_roll_1) >= abs(g_pitch_1) else "pitch"
    lr_euler = "pitch" if abs(g_pitch_2) >= abs(g_roll_2) else "roll"

    fb_gain = g_roll_1 if fb_euler == "roll" else g_pitch_1
    lr_gain = g_pitch_2 if lr_euler == "pitch" else g_roll_2

    return {
        "front_back_euler": fb_euler,
        "left_right_euler": lr_euler,
        "front_back_sign": 1 if fb_gain >= 0 else -1,
        "left_right_sign": 1 if lr_gain >= 0 else -1,
        "gain_deg_per_offset": {
            "front_back": abs(fb_gain),
            "left_right": abs(lr_gain),
        },
        "gain_matrix_roll_pitch": G,
    }


# ---------------------------------------------------------------------------
# 标定流程
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    mocap = MockMocap() if args.mock else MocapReader(args.ip, args.rb)
    if not mocap.connect():
        logger.error("动捕连接失败")
        return 1
    mocap.start()

    if args.mock:
        bus = MockServoBus()
    elif args.servo_port is None:
        logger.error("真实模式必须指定 --servo-port")
        return 1
    else:
        bus = ServoBus(args.servo_port)
    if not bus.connect():
        return 1

    # 1. 回中位，测基准姿态
    bus.send_pair(0, 0)
    time.sleep(SETTLE_S + 0.5)
    q0 = _measure(mocap)
    if q0 is None:
        logger.error("未收到动捕帧")
        return 1
    logger.info("基准姿态 roll=%.3f pitch=%.3f", q0[0], q0[1])

    # 2. 单轴阶跃激励，采集样本
    samples: list[tuple[int, int, float, float]] = []
    for axis_id in (1, 2):
        for amp in STEP_OFFSETS:
            for s in (+1, -1):
                off = s * amp
                _send_single(bus, axis_id, off)
                time.sleep(HOLD_S)
                q = _measure(mocap)
                if q is None:
                    logger.warning("axis=%d off=%d 无动捕帧，跳过", axis_id, off)
                    continue
                droll = q[0] - q0[0]
                dpitch = q[1] - q0[1]
                samples.append((axis_id, off, droll, dpitch))
                logger.info("axis=%d off=%+4d → droll=%+.3f dpitch=%+.3f",
                            axis_id, off, droll, dpitch)
                _send_single(bus, axis_id, 0)
                time.sleep(SETTLE_S)

    # 3. 拟合 + 推导
    if not samples:
        logger.error("无有效样本")
        return 1
    G = fit_gain_matrix(samples)
    calib = derive_calibration(G)
    calib["num_samples"] = len(samples)
    calib["step_offsets"] = STEP_OFFSETS

    # 4. 输出 JSON
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("标定结果: %s", out)

    # 5. 可选 CSV 落盘
    if args.csv:
        _write_samples_csv(Path(args.csv), samples)

    # 6. 收尾回中位
    bus.send_pair(0, 0)
    bus.close()
    mocap.stop()

    _print_result(calib)
    return 0


def _measure(mocap):
    """读动捕，返回 (roll, pitch)（度）。"""
    pose = mocap.get_pose()
    if pose is None:
        return None
    return (pose.roll, pose.pitch)


def _send_single(bus, axis_id: int, offset: int) -> None:
    if axis_id == 1:
        bus.send_pair(offset, 0)
    else:
        bus.send_pair(0, offset)


def _write_samples_csv(path: Path, samples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis_id", "offset", "droll_deg", "dpitch_deg"])
        for row in samples:
            w.writerow(row)


def _print_result(calib: dict) -> None:
    g = calib["gain_deg_per_offset"]
    print("=" * 50)
    print("标定结果")
    print(f"  前后 ← {calib['front_back_euler']} (sign={calib['front_back_sign']:+d}, "
          f"增益={g['front_back']:.4f} °/offset)")
    print(f"  左右 ← {calib['left_right_euler']} (sign={calib['left_right_sign']:+d}, "
          f"增益={g['left_right']:.4f} °/offset)")
    print(f"  增益矩阵 Δ[roll,pitch] = G·Δ[offset_1,offset_2]:")
    for row in calib["gain_matrix_roll_pitch"]:
        print(f"    [{row[0]:+.5f}, {row[1]:+.5f}]")
    print("=" * 50)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
