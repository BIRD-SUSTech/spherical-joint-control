"""开环激励信号（差分 offset 空间，L2 激励层）。

三类判别性激励（设计文档 §9.5）：
    - steps：微幅阶跃 → 暴露静摩擦死区
    - triangle：往返三角波 → 暴露换向迟滞
    - lissajous：低速大摆幅 2D 织网 → 覆盖工作空间与速度区间

幅度以"关节角(度)"给出，内部用 M3 标定增益换算 offset：
    offset = deg / gain，fb 增益 0.0357、lr 增益 0.0428（calibrations/rig1.json）。
"""

from __future__ import annotations

import math

GAIN_FB = 0.0357   # °/offset（M3 标定，前后）
GAIN_LR = 0.0428   # °/offset（M3 标定，左右）


def deg_to_offset(deg: float, axis: str) -> float:
    gain = GAIN_FB if axis == "fb" else GAIN_LR
    return deg / gain


def lissajous(amp_fb_deg, amp_lr_deg, f1, f2, fs, duration_s, phase=0.0):
    """平滑 Lissajous：双轴异频，从零差分起。返回 (t, fb, lr)。"""
    t, fb, lr = [], [], []
    n = int(duration_s * fs)
    for i in range(n):
        ti = i / fs
        t.append(ti)
        fb.append(deg_to_offset(amp_fb_deg * math.sin(2 * math.pi * f1 * ti), "fb"))
        lr.append(deg_to_offset(amp_lr_deg * math.sin(2 * math.pi * f2 * ti + phase), "lr"))
    return t, fb, lr


def _tri_wave(phase: float) -> float:
    """0→1 相位的三角波，从 0 起：0→+1→-1→0。"""
    if phase < 0.25:
        return phase / 0.25
    if phase < 0.75:
        return 1.0 - (phase - 0.25) / 0.25
    return -1.0 + (phase - 0.75) / 0.25


def triangle(axis: str, amp_deg: float, freq: float, fs: float, duration_s: float):
    """单轴往返三角波（恒定速度、换向清晰），暴露迟滞。返回 (t, fb, lr)。"""
    t, fb, lr = [], [], []
    n = int(duration_s * fs)
    period = 1.0 / freq
    for i in range(n):
        ti = i / fs
        tri = amp_deg * _tri_wave((ti % period) / period)
        t.append(ti)
        if axis == "fb":
            fb.append(deg_to_offset(tri, "fb"))
            lr.append(0.0)
        else:
            fb.append(0.0)
            lr.append(deg_to_offset(tri, "lr"))
    return t, fb, lr


def steps(axis: str, amps_deg, hold_s: float, settle_s: float, fs: float):
    """单轴微幅阶跃序列：每个幅度正负各一次，步间回中位。返回 (t, fb, lr)。"""
    t, fb, lr = [], [], []
    ti = 0.0
    for amp in amps_deg:
        for s in (+1, -1):
            off = deg_to_offset(amp * s, axis)
            for _ in range(int(hold_s * fs)):
                t.append(ti)
                fb.append(off if axis == "fb" else 0.0)
                lr.append(off if axis == "lr" else 0.0)
                ti += 1.0 / fs
            for _ in range(int(settle_s * fs)):
                t.append(ti)
                fb.append(0.0)
                lr.append(0.0)
                ti += 1.0 / fs
    return t, fb, lr


def default_segments():
    """默认开环激励段（面向 ±5° 起步，滚雪球后续扩大）。"""
    return [
        {"kind": "steps", "axis": "fb", "amps_deg": [1.0, 2.0], "hold_s": 3.0, "settle_s": 1.0},
        {"kind": "steps", "axis": "lr", "amps_deg": [1.0, 2.0], "hold_s": 3.0, "settle_s": 1.0},
        {"kind": "triangle", "axis": "fb", "amp_deg": 5.0, "freq": 0.1, "duration_s": 20.0},
        {"kind": "triangle", "axis": "lr", "amp_deg": 5.0, "freq": 0.1, "duration_s": 20.0},
        {"kind": "lissajous", "amp_fb_deg": 5.0, "amp_lr_deg": 5.0,
         "f1": 0.1, "f2": 0.16, "duration_s": 30.0},
    ]


def sample_segment(seg: dict, fs: float):
    """展开一条激励段 → (t, fb, lr)。seg 见 default_segments()。"""
    kind = seg["kind"]
    if kind == "lissajous":
        return lissajous(seg["amp_fb_deg"], seg["amp_lr_deg"], seg["f1"], seg["f2"],
                         fs, seg["duration_s"])
    if kind == "triangle":
        return triangle(seg["axis"], seg["amp_deg"], seg["freq"], fs, seg["duration_s"])
    if kind == "steps":
        return steps(seg["axis"], seg["amps_deg"], seg["hold_s"], seg["settle_s"], fs)
    raise ValueError(f"unknown segment kind: {kind}")
