"""开环激励信号（差分 offset 空间，L2 激励层）。

三类判别性激励（设计文档 §9.5）：
    - steps：微幅阶跃 → 暴露静摩擦死区
    - triangle：往返三角波 → 暴露换向迟滞
    - lissajous：低速大摆幅 2D 织网 → 覆盖工作空间与速度区间

幅度以"关节角(度)"给出，内部用增益换算 offset：offset = deg / gain。
增益缺省用 M3 标定值（calibrations/rig1.json），可由调用方传入覆盖（单一事实源）。
"""

from __future__ import annotations

import math

GAIN_FB = 0.057    # °/offset（M4 开环实测 ±5° 有效增益，前后）
GAIN_LR = 0.059    # °/offset（M4 开环实测 ±5° 有效增益，左右）


def deg_to_offset(deg: float, axis: str, gain_fb: float = GAIN_FB, gain_lr: float = GAIN_LR) -> float:
    gain = gain_fb if axis == "fb" else gain_lr
    return deg / gain


def lissajous(amp_fb_deg, amp_lr_deg, f1, f2, fs, duration_s, phase=0.0,
              gain_fb=GAIN_FB, gain_lr=GAIN_LR):
    """平滑 Lissajous：双轴异频，从零差分起。返回 (t, fb, lr)。"""
    t, fb, lr = [], [], []
    n = int(duration_s * fs)
    for i in range(n):
        ti = i / fs
        t.append(ti)
        fb.append(deg_to_offset(amp_fb_deg * math.sin(2 * math.pi * f1 * ti), "fb", gain_fb, gain_lr))
        lr.append(deg_to_offset(amp_lr_deg * math.sin(2 * math.pi * f2 * ti + phase), "lr", gain_fb, gain_lr))
    return t, fb, lr


def _tri_wave(phase: float) -> float:
    """0→1 相位的三角波，从 0 起：0→+1→-1→0。"""
    if phase < 0.25:
        return phase / 0.25
    if phase < 0.75:
        return 1.0 - (phase - 0.25) / 0.25
    return -1.0 + (phase - 0.75) / 0.25


def triangle(axis: str, amp_deg: float, freq: float, fs: float, duration_s: float,
             gain_fb=GAIN_FB, gain_lr=GAIN_LR):
    """单轴往返三角波（恒定速度、换向清晰），暴露迟滞。返回 (t, fb, lr)。"""
    t, fb, lr = [], [], []
    n = int(duration_s * fs)
    period = 1.0 / freq
    for i in range(n):
        ti = i / fs
        tri = amp_deg * _tri_wave((ti % period) / period)
        t.append(ti)
        if axis == "fb":
            fb.append(deg_to_offset(tri, "fb", gain_fb, gain_lr))
            lr.append(0.0)
        else:
            fb.append(0.0)
            lr.append(deg_to_offset(tri, "lr", gain_fb, gain_lr))
    return t, fb, lr


def steps(axis: str, amps_deg, hold_s: float, settle_s: float, fs: float,
          gain_fb=GAIN_FB, gain_lr=GAIN_LR):
    """单轴微幅阶跃序列：每个幅度正负各一次，步间回中位。返回 (t, fb, lr)。"""
    t, fb, lr = [], [], []
    ti = 0.0
    for amp in amps_deg:
        for s in (+1, -1):
            off = deg_to_offset(amp * s, axis, gain_fb, gain_lr)
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


def extended_segments():
    """补数据激励段（M5 充分数据采集，批次 A + C，约 228s）。

    目的（对应 M5 残差结论的补强）：
        - 大角度 ±10° → 增益非线性（滚雪球第一步）
        - 高频 0.3/0.5Hz → 动态/相位滞后
        - 多幅度阶跃 → 死区/增益曲线
        - 重复 M4 关键段 → 一致性验证（用户对单会话结论不信任）
    幅度用 M4 有效增益 0.057/0.059 反算，guardian 70° 兜底。
    """
    return [
        # 批次 A：大角度
        {"kind": "triangle", "axis": "fb", "amp_deg": 10.0, "freq": 0.1, "duration_s": 20.0},
        {"kind": "triangle", "axis": "lr", "amp_deg": 10.0, "freq": 0.1, "duration_s": 20.0},
        # 批次 A：高频
        {"kind": "triangle", "axis": "fb", "amp_deg": 5.0, "freq": 0.3, "duration_s": 20.0},
        {"kind": "triangle", "axis": "fb", "amp_deg": 5.0, "freq": 0.5, "duration_s": 20.0},
        {"kind": "triangle", "axis": "lr", "amp_deg": 5.0, "freq": 0.3, "duration_s": 20.0},
        # 批次 A：大角度 2D 覆盖
        {"kind": "lissajous", "amp_fb_deg": 10.0, "amp_lr_deg": 10.0,
         "f1": 0.1, "f2": 0.16, "duration_s": 30.0},
        # 批次 A：多幅度阶跃（死区/增益曲线）
        {"kind": "steps", "axis": "fb", "amps_deg": [1.0, 2.0, 5.0], "hold_s": 3.0, "settle_s": 1.0},
        {"kind": "steps", "axis": "lr", "amps_deg": [1.0, 2.0, 5.0], "hold_s": 3.0, "settle_s": 1.0},
        # 批次 C：重复性（与 M4 关键段一致，验证一致性）
        {"kind": "triangle", "axis": "fb", "amp_deg": 5.0, "freq": 0.1, "duration_s": 20.0},
        {"kind": "lissajous", "amp_fb_deg": 5.0, "amp_lr_deg": 5.0,
         "f1": 0.1, "f2": 0.16, "duration_s": 30.0},
    ]


def sample_segment(seg: dict, fs: float, gain_fb: float = GAIN_FB, gain_lr: float = GAIN_LR):
    """展开一条激励段 → (t, fb, lr)。seg 见 default_segments()。"""
    kind = seg["kind"]
    if kind == "lissajous":
        return lissajous(seg["amp_fb_deg"], seg["amp_lr_deg"], seg["f1"], seg["f2"],
                         fs, seg["duration_s"], gain_fb=gain_fb, gain_lr=gain_lr)
    if kind == "triangle":
        return triangle(seg["axis"], seg["amp_deg"], seg["freq"], fs, seg["duration_s"],
                        gain_fb=gain_fb, gain_lr=gain_lr)
    if kind == "steps":
        return steps(seg["axis"], seg["amps_deg"], seg["hold_s"], seg["settle_s"], fs,
                     gain_fb=gain_fb, gain_lr=gain_lr)
    raise ValueError(f"unknown segment kind: {kind}")
