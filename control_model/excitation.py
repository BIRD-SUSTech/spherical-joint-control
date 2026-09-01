"""开环激励生成器（无 IK，直接在舵机差分/共模空间设计）。

坐标约定（归一化舵机角 n ∈ [-1, 1]，0 = 中位）：
    d1 = n1 - n3     # pitch 对抗对差分（驱动 pitch）
    d2 = n2 - n4     # yaw   对抗对差分（驱动 yaw）
    p  = 全局共模预紧（冗余自由度，只负责绷紧缆绳）

解码：
    n = [p + d1/2, p + d2/2, p - d1/2, p - d2/2]   （逐轴夹到 [-1, 1]）

设计要点（针对大且不确定的死区/静摩擦）：激励幅度要"足够大"以保证真实系统
流畅滑动，而不是小幅扫频。因此默认给出低速、大摆幅、平滑的 Lissajous 轨迹，
且各段从零差分起、平滑进入，避免段首阶跃。
"""

from __future__ import annotations

import numpy as np


def decode(d1, d2, p):
    """差分/共模 -> 4 个归一化舵机角 [-1, 1]。"""
    n = np.array([p + d1 / 2.0, p + d2 / 2.0, p - d1 / 2.0, p - d2 / 2.0])
    return np.clip(n, -1.0, 1.0)


def d_from_joint_deg(deg, direct_gain=0.01):
    """期望关节角(度) -> 差分幅度(归一化)。

    差分 d = 单侧偏置 * 2 = 2 * direct_gain * deg。
    direct_gain 用 data_collection/scripts/calibrate_direct_gain.py 实测，
    缺省 0.01（约 1° 关节角对应 0.01 归一化单侧偏置）。
    """
    return 2.0 * direct_gain * deg


def lissajous(amp, f1, f2, p, fs, duration_s, phase=0.0):
    """平滑 Lissajous：双轴异频、低速大摆幅；d1(0)=d2(0)=0，平滑起步。"""
    t = np.arange(0.0, duration_s, 1.0 / fs)
    d1 = amp * np.sin(2.0 * np.pi * f1 * t)
    d2 = amp * np.sin(2.0 * np.pi * f2 * t + phase)
    return t, decode(d1, d2, p)


def circle(amp, period_s, p, fs, duration_s):
    """等频圆轨迹（pitch-yaw 相位差 90°）。注意 t=0 时非零差分，会有起始阶跃。"""
    t = np.arange(0.0, duration_s, 1.0 / fs)
    d1 = amp * np.cos(2.0 * np.pi * t / period_s)
    d2 = amp * np.sin(2.0 * np.pi * t / period_s)
    return t, decode(d1, d2, p)


def default_segments():
    """默认开环激励段（足够大、足够平滑，保证真实系统流畅运动）。

    幅值 0.40~0.55（差分归一化），在动捕里应看到明显跟随；若实测运动不足，
    等比放大 amp（差分上限 ~0.8 仍安全：p=0.15 时舵机不会越界）。
    """
    return [
        {"kind": "lissajous", "amp": 0.40, "f1": 0.10, "f2": 0.16, "duration_s": 30.0},
        {"kind": "lissajous", "amp": 0.50, "f1": 0.22, "f2": 0.09, "duration_s": 30.0},
        {"kind": "lissajous", "amp": 0.55, "f1": 0.14, "f2": 0.30, "duration_s": 30.0},
    ]


def sample_segment(seg, p, fs):
    """展开一条激励段 -> (t_s, u (4, N) 归一化)。seg 见 default_segments()。"""
    kind = seg["kind"]
    if kind == "lissajous":
        return lissajous(seg["amp"], seg["f1"], seg["f2"], p, fs, seg["duration_s"])
    if kind == "circle":
        return circle(seg["amp"], seg["period_s"], p, fs, seg["duration_s"])
    raise ValueError(f"unknown excitation kind: {kind}")
