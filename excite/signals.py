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

GAIN_FB = 0.057    # ⚠️ rig1(旧硬件)遗留默认，仅无标定时占位；实机由 calibrations/*.json 覆盖
GAIN_LR = 0.059    # ⚠️ 新硬件(rig2)舵机更强、真实增益更大，开环必须用 M3 实测增益，勿用此值


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


def large_angle_segments():
    """大角度滚雪球激励段（±15°/±20°，补增益非线性外推数据）。

    M10 发现变速圆 +12° 外推有 +2° 残差（g(q) 只拟合到 ±10°）。
    本段集用更低频（大摆幅 + 低速）覆盖 ±15°/±20°，guardian 70° 兜底。
    """
    return [
        {"kind": "triangle", "axis": "fb", "amp_deg": 15.0, "freq": 0.08, "duration_s": 25.0},
        {"kind": "triangle", "axis": "lr", "amp_deg": 15.0, "freq": 0.08, "duration_s": 25.0},
        {"kind": "triangle", "axis": "fb", "amp_deg": 20.0, "freq": 0.06, "duration_s": 30.0},
        {"kind": "triangle", "axis": "lr", "amp_deg": 20.0, "freq": 0.06, "duration_s": 30.0},
        {"kind": "lissajous", "amp_fb_deg": 15.0, "amp_lr_deg": 15.0,
         "f1": 0.08, "f2": 0.12, "duration_s": 40.0},
        {"kind": "steps", "axis": "fb", "amps_deg": [5.0, 10.0, 15.0], "hold_s": 4.0, "settle_s": 2.0},
        {"kind": "steps", "axis": "lr", "amps_deg": [5.0, 10.0, 15.0], "hold_s": 4.0, "settle_s": 2.0},
    ]


def extreme_segments():
    """40° 工作空间扩展段（±30°/±40°，渐进，滚雪球最后一步）。

    ⚠️ 安全前提：增益随幅度爬升（旧 rig ±1°→±5° 已 1.6×），用小幅增益直接换算 40°
    的 offset 会超调、逼近 guardian 70°。因此本段集【必须】在 ±15/20° 段测出割线增益、
    用 model.measure_gain 校正后再跑（见 current_stage_cmds.txt 滚雪球节）。
    本段集用更低频（大摆幅 + 低速），guardian 70° 兜底。
    """
    return [
        {"kind": "triangle", "axis": "fb", "amp_deg": 30.0, "freq": 0.05, "duration_s": 40.0},
        {"kind": "triangle", "axis": "lr", "amp_deg": 30.0, "freq": 0.05, "duration_s": 40.0},
        {"kind": "triangle", "axis": "fb", "amp_deg": 40.0, "freq": 0.04, "duration_s": 50.0},
        {"kind": "triangle", "axis": "lr", "amp_deg": 40.0, "freq": 0.04, "duration_s": 50.0},
        {"kind": "lissajous", "amp_fb_deg": 30.0, "amp_lr_deg": 30.0,
         "f1": 0.04, "f2": 0.06, "duration_s": 60.0},
        {"kind": "steps", "axis": "fb", "amps_deg": [10.0, 20.0, 30.0, 40.0],
         "hold_s": 5.0, "settle_s": 2.0},
        {"kind": "steps", "axis": "lr", "amps_deg": [10.0, 20.0, 30.0, 40.0],
         "hold_s": 5.0, "settle_s": 2.0},
    ]


def multisine(amp_fb_deg, amp_lr_deg, f_lo, f_hi, n_tone, duration_s, fs, seed=0,
              gain_fb=GAIN_FB, gain_lr=GAIN_LR, fade_s=1.5):
    """带限随机相位多正弦（**辨识专用**激励）。返回 (t, fb, lr)。

    为什么必须用它：三角波/Lissajous/扫频都是【平滑可外推】的——模型能从状态历史里
    把 u 外推出来，于是 (历史, u_t) 相对历史【零信息增量】，∂F/∂u 不可辨识（实测：
    去掉 u 输入后留出 RMSE 不变）。随机相位多正弦在 u 空间是**外生、宽带、不可外推**的，
    是系统辨识的标准激励（Schoukens 多正弦）。

    安全性：生成后按 **峰值** 归一到 amp_deg，再乘 raised-cosine 淡入；
    故 offset 峰值 = amp_deg/gain，可由调用方控制（guardian 70° 兜底）。
    双轴用独立随机相位 → 轴间不相关，2×2 增益阵可辨识。
    """
    import random
    rng = random.Random(seed)
    n = int(duration_s * fs)
    # 对数均匀分布频率点，覆盖关注频带
    if n_tone == 1:
        freqs = [f_lo]
    else:
        freqs = [f_lo * (f_hi / f_lo) ** (k / (n_tone - 1)) for k in range(n_tone)]

    def _wave(phases):
        w = [0.0] * n
        for f, ph in zip(freqs, phases):
            for i in range(n):
                w[i] += math.sin(2 * math.pi * f * (i / fs) + ph)
        return w

    wf = _wave([rng.uniform(0, 2 * math.pi) for _ in freqs])
    wl = _wave([rng.uniform(0, 2 * math.pi) for _ in freqs])
    # 峰值归一 + raised-cosine 淡入（从 0 起，避免上电跳变）
    mf = max(abs(x) for x in wf) or 1.0
    ml = max(abs(x) for x in wl) or 1.0
    nf = int(fade_s * fs)
    t, fb, lr = [], [], []
    for i in range(n):
        g = 1.0 if i >= nf else 0.5 * (1 - math.cos(math.pi * i / max(nf, 1)))
        t.append(i / fs)
        fb.append(deg_to_offset(amp_fb_deg * wf[i] / mf * g, "fb", gain_fb, gain_lr))
        lr.append(deg_to_offset(amp_lr_deg * wl[i] / ml * g, "lr", gain_fb, gain_lr))
    return t, fb, lr


def ident_segments():
    """**辨识专用**激励段集：u 外生宽带随机激励（反解 F 模式的前提）。

    与 triangle/lissajous/steps 段集的本质区别：那些段是平滑可外推的，模型能仅凭
    状态历史预测 u → (历史, u) 无信息增量 → ∂F/∂u 不可辨识。本段集用随机相位多正弦
    直接驱动 offset，使 u 在统计上**独立于状态历史**。

    预算 ~240s：3 个幅度档 × 2 个种子（覆盖增益非线性 + 可重复性），
    峰值 offset = amp/gain，按 ±10/±15/±20° 三档，远低于 guardian 70°。
    """
    return [
        {"kind": "multisine", "amp_fb_deg": 10.0, "amp_lr_deg": 10.0,
         "f_lo": 0.05, "f_hi": 2.0, "n_tone": 24, "duration_s": 40.0, "seed": 1},
        {"kind": "multisine", "amp_fb_deg": 15.0, "amp_lr_deg": 15.0,
         "f_lo": 0.05, "f_hi": 2.0, "n_tone": 24, "duration_s": 40.0, "seed": 2},
        {"kind": "multisine", "amp_fb_deg": 20.0, "amp_lr_deg": 20.0,
         "f_lo": 0.05, "f_hi": 2.0, "n_tone": 24, "duration_s": 40.0, "seed": 3},
        {"kind": "multisine", "amp_fb_deg": 15.0, "amp_lr_deg": 15.0,
         "f_lo": 0.05, "f_hi": 2.0, "n_tone": 24, "duration_s": 40.0, "seed": 7},
        {"kind": "multisine", "amp_fb_deg": 15.0, "amp_lr_deg": 15.0,
         "f_lo": 0.05, "f_hi": 2.0, "n_tone": 24, "duration_s": 40.0, "seed": 11},
        {"kind": "multisine", "amp_fb_deg": 10.0, "amp_lr_deg": 10.0,
         "f_lo": 0.3, "f_hi": 5.0, "n_tone": 20, "duration_s": 40.0, "seed": 5},
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
    if kind == "multisine":
        return multisine(seg["amp_fb_deg"], seg["amp_lr_deg"], seg["f_lo"], seg["f_hi"],
                         seg["n_tone"], seg["duration_s"], fs, seg.get("seed", 0),
                         gain_fb=gain_fb, gain_lr=gain_lr)
    raise ValueError(f"unknown segment kind: {kind}")
