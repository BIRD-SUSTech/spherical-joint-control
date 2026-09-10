"""诊断扫描：开环静态四缆张力 T_i(q_orth) vs 正交侧角度曲线。

目的：精确回答"哪根缆在哪个角度掉到 ≈0（松弛）"。与 calibrate_co_tension 的差别：
    - 不找 c_needed，而是直接记录每个角度下的 ch1..ch4 张力（中位 + 最小值）；
    - 更密的角度网格（默认 ±40° 每 5°）；
    - 落盘 CSV，事后可画 T(q) 曲线。

做法：发单轴恒定差分 offset（正交轴 u=0 = 换向差分条件），等自由响应衰减，
读动捕实际位姿 + 力传感器 ch1-ch4。

用法：
    python -m control.scan_tension --servo-port COM5 --force-port COM3 \
        --calibration calibrations/rig2.json --gain-fb 0.090 --gain-lr 0.082 \
        --min-deg -40 --max-deg 40 --step-deg 5 --out collect/logs/tension_vs_angle.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

from control.calibration import Calibration
from excite.guardian import Guardian
from hardware.mocap import MockMocap, MocapReader
from hardware.servo import MockServoBus, ServoBus

logger = logging.getLogger(__name__)

FORCE_SLAVE = 0x01
FORCE_REG_START = 0x000B
FORCE_CHANNELS = 6

FIELDS = ["axis", "target_deg", "offset", "q_fb", "q_lr",
          "ch1_med", "ch2_med", "ch3_med", "ch4_med",
          "ch1_min", "ch2_min", "ch3_min", "ch4_min", "co_mode"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="开环静态四缆张力 vs 角度扫描")
    p.add_argument("--mock", action="store_true", help="离线自检（合成动捕/力）")
    p.add_argument("--servo-port", default=None, help="舵机串口（真实模式必填）")
    p.add_argument("--force-port", default=None, help="力传感器串口（真实模式必填）")
    p.add_argument("--ip", default="10.1.1.198", help="动捕服务器 IP")
    p.add_argument("--rb", type=int, default=0, help="动捕刚体索引")
    p.add_argument("--calibration", required=True, help="标定 JSON（需 gain_deg_per_offset）")
    p.add_argument("--gain-fb", type=float, default=None, help="fb 轴开环换算增益覆盖（°/offset）")
    p.add_argument("--gain-lr", type=float, default=None, help="lr 轴开环换算增益覆盖（°/offset）")
    p.add_argument("--max-offset", type=float, default=500.0, help="单轴 offset 上限（防增益爬升超调）")
    p.add_argument("--min-deg", type=float, default=-40.0, help="扫描角度下限")
    p.add_argument("--max-deg", type=float, default=40.0, help="扫描角度上限")
    p.add_argument("--step-deg", type=float, default=5.0, help="扫描角度步长")
    p.add_argument("--hold-s", type=float, default=4.0, help="每点开环保持时长 s（等自由响应衰减）")
    p.add_argument("--n-force", type=int, default=6, help="每点力采样次数（0.1s 间隔）")
    p.add_argument("--co-mode", type=float, default=0.0,
                   help="施加在【正交对】上的共模预紧 offset（默认 0 = 中性，看自然松弛）")
    p.add_argument("--out", default="collect/logs/tension_vs_angle.csv", help="输出 CSV")
    p.add_argument("--limit-deg", type=float, default=70.0, help="guardian 限位")
    return p.parse_args()


class _MockForce:
    def read_32bit_values(self, slave, start, count, signed=True):
        return [3000] * count


def _read_force(client) -> np.ndarray:
    """读 ch1..ch4 张力（3 样本中位）。"""
    samples = []
    for _ in range(3):
        raw = client.read_32bit_values(FORCE_SLAVE, FORCE_REG_START, FORCE_CHANNELS, signed=True)
        samples.append(raw[:4])
        time.sleep(0.02)
    return np.median(np.asarray(samples, dtype=float), axis=0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    calib = Calibration.load(args.calibration)
    if not calib.gain_deg_per_offset:
        logger.error("标定缺 gain_deg_per_offset；先跑 M3")
        return 1
    g_fb = args.gain_fb if args.gain_fb is not None else calib.gain_deg_per_offset["front_back"]
    g_lr = args.gain_lr if args.gain_lr is not None else calib.gain_deg_per_offset["left_right"]

    mocap = MockMocap() if args.mock else MocapReader(args.ip, args.rb)
    if not mocap.connect():
        logger.error("动捕连接失败")
        return 1
    mocap.start()
    if not args.mock:
        deadline = time.time() + 3.0
        while time.time() < deadline and mocap.get_pose() is None:
            time.sleep(0.01)
        if mocap.get_pose() is None:
            logger.error("3s 内未收到动捕帧")
            return 1

    if args.mock:
        bus = MockServoBus()
    elif args.servo_port is None:
        logger.error("真实模式必须指定 --servo-port")
        return 1
    else:
        bus = ServoBus(args.servo_port)
    if not bus.connect():
        return 1

    if args.mock:
        force_client = _MockForce()
    elif args.force_port is None:
        logger.error("真实模式必须指定 --force-port")
        return 1
    else:
        from collect.sensors.modbus import ModbusClient
        force_client = ModbusClient(port=args.force_port, baudrate=19200, timeout=0.5)

    guardian = Guardian(mocap, bus, calib, args.limit_deg)
    guardian.start()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()

    angles = np.arange(args.min_deg, args.max_deg + 1e-9, args.step_deg)
    try:
        for axis, gain in (("fb", g_fb), ("lr", g_lr)):
            logger.info("=== 扫描 %s 轴（正交对共模 c=%.0f）===", axis, args.co_mode)
            for target in angles:
                if guardian.is_triggered:
                    logger.error("guardian 触发，中止")
                    break
                off = target / gain if gain else 0.0
                off = max(-args.max_offset, min(args.max_offset, off))
                u_fb, u_lr = (off, 0.0) if axis == "fb" else (0.0, off)
                # 正交对共模：fb 轴驱动时正交对是 lr，lr 轴驱动时正交对是 fb
                c_fb, c_lr = (0.0, args.co_mode) if axis == "fb" else (args.co_mode, 0.0)
                # 开环保持 hold_s 秒（自由响应衰减）
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < args.hold_s:
                    if guardian.is_triggered:
                        break
                    bus.send_pair_tension(int(u_fb), int(u_lr), int(c_fb), int(c_lr))
                    time.sleep(0.01)
                # 读力（n 次，0.1s 间隔，得中位 + 最小）
                ch = []
                for _ in range(args.n_force):
                    if guardian.is_triggered:
                        break
                    ch.append(_read_force(force_client))
                    time.sleep(0.1)
                ch = np.asarray(ch)
                pose = mocap.get_pose()
                q_fb = q_lr = float("nan")
                if pose is not None:
                    q_fb, q_lr = calib.map_pose(pose.roll, pose.pitch)
                row = {
                    "axis": axis, "target_deg": round(float(target), 1),
                    "offset": round(off, 1),
                    "q_fb": round(float(q_fb), 2), "q_lr": round(float(q_lr), 2),
                    "co_mode": args.co_mode,
                }
                if len(ch):
                    med = np.median(ch, axis=0)
                    mn = np.min(ch, axis=0)
                    for i in range(4):
                        row[f"ch{i+1}_med"] = round(float(med[i]), 0)
                        row[f"ch{i+1}_min"] = round(float(mn[i]), 0)
                writer.writerow(row)
                fh.flush()
                med = np.median(ch, axis=0)
                mn = np.min(ch, axis=0)
                logger.info("  %s target=%+.0f° off=%+.0f → q=(%+.1f,%+.1f) ch_med=[%s] ch_min=[%s]",
                            axis, target, off, q_fb, q_lr,
                            " ".join("%d" % int(v) for v in med),
                            " ".join("%d" % int(v) for v in mn))
            # 回中位
            bus.send_pair_tension(0, 0, 0, 0)
            time.sleep(1.5)
    finally:
        bus.send_pair_tension(0, 0, 0, 0)
        fh.close()
        guardian.stop()
        guardian.join(timeout=2.0)
        bus.close()
        mocap.stop()
        if not args.mock:
            force_client.close()
    logger.info("张力扫描完成 → %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
