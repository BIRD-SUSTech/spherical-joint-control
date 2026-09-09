"""共模预紧标定（§8.3.2）：测 c(q_orth) 偶函数，数据驱动、不猜。

背景：大角度开环/闭环实机已证——±20° 起四缆张力掉到 ≈0（完全松弛），是"对称几何松弛"
（∝q_orth²），差分前馈（gain_cross 奇项）补不了，必须共模预紧。本脚本：
    1. 闭环把球杆钉在一系列位姿（沿一轴 +0/±20/±40°）；
    2. 在该位姿上对【正交对】两缆同加共模 offset c，读力测 min(两缆张力)；
    3. 找使 min 张力 ≥ T_min 的最小 c，得 (q_orth, c_needed) 样本；
    4. 拟合偶多项式 c(q_orth)=c0+c2·q²+…，写回标定 JSON 的 co_tension 字段。

写入后 `Calibration.common_mode()` 自动生效（闭环/开环发送链路已接，gated on has_co_tension）。

用法：
    python -m control.calibrate_co_tension \
        --servo-port COM5 --force-port COM3 \
        --calibration calibrations/rig2.json \
        --controller-config configs/rig2_v2.json \
        --t-min 500 --c-max 400 --c-step 50
    # 默认把 co_tension 合并写回 --calibration；--out 可另存。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

from control.calibration import Calibration
from control.controller_config import ControllerConfig
from control.pid import PIDController
from excite.guardian import Guardian
from hardware.mocap import MockMocap, MocapReader
from hardware.servo import MockServoBus, ServoBus

logger = logging.getLogger(__name__)

# 每对的标定位姿 (target_fb, target_lr)；正交轴决定 c 的自变量
# front_back 对：c_fb(q_lr)，故沿 lr 轴扫（fb=0）
# left_right 对：c_lr(q_fb)，故沿 fb 轴扫（lr=0）
PAIR_POSES = {
    "front_back": [(0.0, 0.0), (0.0, 20.0), (0.0, -20.0), (0.0, 40.0), (0.0, -40.0)],
    "left_right": [(0.0, 0.0), (20.0, 0.0), (-20.0, 0.0), (40.0, 0.0), (-40.0, 0.0)],
}

# 力通道索引：ch1..ch4 ↔ CH1..CH4（硬件已确认顺序一致）
PAIR_FORCE_IDX = {"front_back": (0, 2), "left_right": (1, 3)}

FORCE_SLAVE = 0x01
FORCE_REG_START = 0x000B
FORCE_CHANNELS = 6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="共模预紧标定 c(q_orth)")
    p.add_argument("--mock", action="store_true", help="离线自检（合成动捕/力，不接硬件）")
    p.add_argument("--servo-port", default=None, help="舵机串口（真实模式必填）")
    p.add_argument("--force-port", default=None, help="力传感器串口（真实模式必填）")
    p.add_argument("--ip", default="10.1.1.198", help="动捕服务器 IP")
    p.add_argument("--rb", type=int, default=0, help="动捕刚体索引")
    p.add_argument("--calibration", required=True, help="标定 JSON（需含 gain_deg_per_offset）")
    p.add_argument("--controller-config", default=None,
                   help="闭环持位用前馈（大角度建议给 rig2_v2 等 g(q)）")
    p.add_argument("--out", default=None, help="输出 JSON（缺省=覆盖 --calibration）")
    p.add_argument("--t-min", type=float, default=500.0, help="张力下限（四缆都 ≥ 此值才够紧）")
    p.add_argument("--c-max", type=float, default=400.0, help="共模 offset 扫描上限")
    p.add_argument("--c-step", type=float, default=50.0, help="共模 offset 扫描步长")
    p.add_argument("--hold-s", type=float, default=3.0, help="每个位姿闭环持位时长 s")
    p.add_argument("--sweep-s", type=float, default=0.5, help="每个 c 值保持时长 s（等张力稳定）")
    p.add_argument("--settle-s", type=float, default=1.5, help="对间回中位时长 s")
    p.add_argument("--degree", type=int, default=2, help="偶多项式阶数（2 → c0+c2·q²）")
    p.add_argument("--dynamic", action="store_true",
                   help="动态标定：跑驱动轴正弦(制造换向)，扫【恒定】共模 c0，测动态最小张力（推荐）")
    p.add_argument("--drive-amp", type=float, default=20.0, help="动态标定驱动轴幅度 °")
    p.add_argument("--drive-freq", type=float, default=0.1, help="动态标定驱动轴频率 Hz")
    p.add_argument("--drive-duration", type=float, default=20.0, help="每个 c0 值跑轨迹时长 s")
    p.add_argument("--kp", type=float, default=8.0)
    p.add_argument("--ki", type=float, default=1.0)
    p.add_argument("--kd", type=float, default=2.5)
    p.add_argument("--deadband", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--limit-deg", type=float, default=70.0, help="guardian 限位")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 力读取（同步，3 样本取中位）
# ---------------------------------------------------------------------------

class _MockForce:
    def read_32bit_values(self, slave, start, count, signed=True):
        return [3000] * count  # 恒张力：mock 下 c_needed=0，验证流程跑通


def _read_force(client) -> np.ndarray:
    """读 ch1..ch4 张力（3 样本中位），返回 (4,) 数组。"""
    samples = []
    for _ in range(3):
        raw = client.read_32bit_values(FORCE_SLAVE, FORCE_REG_START, FORCE_CHANNELS, signed=True)
        samples.append(raw[:4])
        time.sleep(0.02)
    return np.median(np.asarray(samples, dtype=float), axis=0)


# ---------------------------------------------------------------------------
# 持位 + 共模扫描
# ---------------------------------------------------------------------------

def hold_pose(bus, mocap, calib, pid_fb, pid_lr, controller,
              t_fb, t_lr, hold_s, fs=100.0) -> tuple[float, float]:
    """闭环把球杆钉在 (t_fb, t_lr)，返回稳态差分 offset（共模=0）。"""
    pid_fb.target = t_fb
    pid_lr.target = t_lr
    u_fb = u_lr = 0.0
    n = int(hold_s * fs)
    for _ in range(n):
        pose = mocap.get_pose()
        if pose is None:
            time.sleep(1.0 / fs)
            continue
        curr_fb, curr_lr = calib.map_pose(pose.roll, pose.pitch)
        u_fb = pid_fb.calculate(curr_fb)
        u_lr = pid_lr.calculate(curr_lr)
        if controller.has_feedforward():
            ff_fb, ff_lr = controller.feedforward(t_fb, t_lr)
            u_fb += ff_fb
            u_lr += ff_lr
        bus.send_pair_tension(int(u_fb), int(u_lr), 0, 0)
        time.sleep(1.0 / fs)
    return u_fb, u_lr


def sweep_common(bus, force_client, pair, u_fb, u_lr, c_values, t_min, sweep_s):
    """固定差分 + 对 pair 两缆同加共模 c，找 min 张力 ≥ t_min 的最小 c。"""
    i1, i2 = PAIR_FORCE_IDX[pair]
    for c in c_values:
        c_fb = c if pair == "front_back" else 0
        c_lr = c if pair == "left_right" else 0
        bus.send_pair_tension(int(u_fb), int(u_lr), int(c_fb), int(c_lr))
        time.sleep(sweep_s)
        t = _read_force(force_client)
        if min(t[i1], t[i2]) >= t_min:
            return c
    return c_values[-1]  # 饱和：c_max 仍未达 t_min


def fit_even(q_abs: np.ndarray, c: np.ndarray, degree: int) -> np.ndarray:
    """偶多项式 c(q)=c0+c2·q²+…，返回 [c0, c2, c4, ...]。

    degree = 最高偶次指数（2 → c0+c2·q²；4 → 再加 c4·q⁴）。
    """
    n_terms = degree // 2  # 非常数偶次项个数
    cols = [np.ones_like(q_abs)]
    for k in range(1, n_terms + 1):
        cols.append(q_abs ** (2 * k))
    A = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(A, c, rcond=None)
    return coeffs


def dynamic_min_tension(bus, mocap, calib, force_client, pid_fb, pid_lr, controller,
                        pair, amp, freq, duration, c_pair, fs=100.0) -> float:
    """跑驱动轴正弦（正交轴持 0），叠恒定共模 c_pair，返回该对最小张力。

    pair="left_right"（ch2/ch4）松弛由 fb 倾斜造成 → 驱动 fb 轴；
    pair="front_back"（ch1/ch3）松弛由 lr 倾斜造成 → 驱动 lr 轴。
    正弦在峰值处换向 = 最大倾斜 + 换向重合，正是动态松弛最深处。
    """
    i1, i2 = PAIR_FORCE_IDX[pair]
    drive_fb = (pair == "left_right")
    tmin = float("inf")
    t0 = time.perf_counter()
    last_force = time.time() - 1.0
    pid_fb.reset()
    pid_lr.reset()
    while time.perf_counter() - t0 < duration:
        t = time.perf_counter() - t0
        drive = amp * math.sin(2 * math.pi * freq * t)
        t_fb = drive if drive_fb else 0.0
        t_lr = 0.0 if drive_fb else drive
        pose = mocap.get_pose()
        if pose is None:
            time.sleep(1.0 / fs)
            continue
        curr_fb, curr_lr = calib.map_pose(pose.roll, pose.pitch)
        pid_fb.target = t_fb
        pid_lr.target = t_lr
        u_fb = pid_fb.calculate(curr_fb)
        u_lr = pid_lr.calculate(curr_lr)
        if controller.has_feedforward():
            ff_fb, ff_lr = controller.feedforward(t_fb, t_lr)
            u_fb += ff_fb
            u_lr += ff_lr
        c_fb = c_pair if pair == "front_back" else 0
        c_lr = c_pair if pair == "left_right" else 0
        bus.send_pair_tension(int(u_fb), int(u_lr), int(c_fb), int(c_lr))
        if time.time() - last_force > 0.1:  # ~10Hz 读力
            tval = _read_force(force_client)
            tmin = min(tmin, tval[i1], tval[i2])
            last_force = time.time()
        time.sleep(1.0 / fs)
    return tmin


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    calib = Calibration.load(args.calibration)
    if not calib.gain_deg_per_offset:
        logger.error("标定缺 gain_deg_per_offset；先跑 M3 标定再标定共模预紧")
        return 1
    controller = (ControllerConfig.load(args.controller_config)
                  if args.controller_config else ControllerConfig.none())

    # 动捕
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

    # 舵机
    if args.mock:
        bus = MockServoBus()
    elif args.servo_port is None:
        logger.error("真实模式必须指定 --servo-port")
        return 1
    else:
        bus = ServoBus(args.servo_port)
    if not bus.connect():
        return 1

    # 力（同步 MODBUS）
    if args.mock:
        force_client = _MockForce()
    elif args.force_port is None:
        logger.error("真实模式必须指定 --force-port")
        return 1
    else:
        from collect.sensors.modbus import ModbusClient
        force_client = ModbusClient(port=args.force_port, baudrate=19200, timeout=0.5)

    # guardian
    guardian = Guardian(mocap, bus, calib, args.limit_deg)
    guardian.start()

    pid_fb = PIDController(kp=args.kp, ki=args.ki, kd=args.kd, limit=400.0,
                           deadband=args.deadband, alpha=args.alpha)
    pid_lr = PIDController(kp=args.kp, ki=args.ki, kd=args.kd, limit=400.0,
                           deadband=args.deadband, alpha=args.alpha)

    c_values = [round(v, 1) for v in np.arange(0.0, args.c_max + 1e-9, args.c_step)]

    co_tension = {}
    if args.dynamic:
        # 动态标定（推荐）：扫【恒定】共模 c0，跑驱动轴正弦制造换向，测动态最小张力。
        # 动态换向松弛是恒定预紧问题，不是 ∝q² 的位置函数；静态持位测不到。
        for pair in ("front_back", "left_right"):
            drive = "fb" if pair == "left_right" else "lr"
            logger.info("=== 动态标定 %s 对（驱动 %s 轴 ±%.0f°@%.2fHz）===",
                        pair, drive, args.drive_amp, args.drive_freq)
            c_needed = args.c_max
            for c0 in c_values:
                if guardian.is_triggered:
                    logger.error("guardian 触发，中止")
                    break
                tmin = dynamic_min_tension(bus, mocap, calib, force_client,
                                           pid_fb, pid_lr, controller,
                                           pair, args.drive_amp, args.drive_freq,
                                           args.drive_duration, c0)
                logger.info("  c0=%3.0f → 最小张力=%.0f", c0, tmin)
                if tmin >= args.t_min:
                    c_needed = c0
                    break
            co_tension[pair] = [float(c_needed)]  # 恒定预紧（无 q² 项）
            bus.send_pair_tension(0, 0, 0, 0)
            time.sleep(args.settle_s)
    else:
        # 静态标定（偶多项式；⚠️ 静态持位测不到动态换向松弛，大角度场景请用 --dynamic）
        samples = {"front_back": [], "left_right": []}
        for pair, poses in PAIR_POSES.items():
            orth_idx = 1 if pair == "front_back" else 0
            logger.info("=== 静态标定 %s 对（正交轴=%s）===", pair, "lr" if orth_idx == 1 else "fb")
            for (t_fb, t_lr) in poses:
                if guardian.is_triggered:
                    logger.error("guardian 触发，中止")
                    break
                q_orth = t_lr if orth_idx == 1 else t_fb
                u_fb, u_lr = hold_pose(bus, mocap, calib, pid_fb, pid_lr, controller,
                                       t_fb, t_lr, args.hold_s)
                c_needed = sweep_common(bus, force_client, pair, u_fb, u_lr,
                                        c_values, args.t_min, args.sweep_s)
                samples[pair].append((abs(q_orth), c_needed))
                sat = "（饱和！c_max 未达 T_min）" if c_needed >= args.c_max else ""
                logger.info("  位姿(fb=%+.0f,lr=%+.0f) |q_orth|=%2.0f° → c=%5.0f%s",
                            t_fb, t_lr, abs(q_orth), c_needed, sat)
            bus.send_pair_tension(0, 0, 0, 0)
            time.sleep(args.settle_s)

        for pair in ("front_back", "left_right"):
            qs = np.array([s[0] for s in samples[pair]])
            cs = np.array([s[1] for s in samples[pair]])
            if len(qs) < 2:
                logger.error("%s 样本不足，无法拟合", pair)
                co_tension[pair] = [0.0] * (args.degree + 1)
                continue
            coeffs = fit_even(qs, cs, args.degree)
            coeffs = np.maximum(coeffs, 0.0)
            co_tension[pair] = [float(c) for c in coeffs]
            logger.info("%s 共模偶多项式 c(q_orth)=%s", pair,
                        "  ".join(f"c{k*2}={c:+.2f}" for k, c in enumerate(coeffs)))

    out = Path(args.out or args.calibration)
    data = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    data["co_tension"] = co_tension
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 收尾回中位 + 释放共模
    bus.send_pair_tension(0, 0, 0, 0)
    time.sleep(1.0)
    bus.close()
    mocap.stop()
    if not args.mock:
        force_client.close()
    logger.info("共模预紧标定完成 → %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
