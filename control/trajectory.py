"""闭环目标轨迹生成（含 ramp-in 缓启动，单一事实源）。

轨迹类型：
    - circle：圆形（等频，相位差 90°）
    - hold：恒定目标
    - lissajous：双频 2D 织网
    - eight：8 字形（1:2 频率比，换向频繁）
    - variable_circle：变速圆（快慢交替，覆盖速度区间）
    - waypoints：点到点（多 hold 目标，大角度驻留）
    - speed_ladder：速度阶梯（同幅度依次跑多频率，覆盖 |q̇| 区间）——§8.4 D0
    - random_fourier：随机多频 Fourier 叠加（(q,q̇) 空间覆盖最大化）——§8.4 D0

所有动态轨迹前 RAMP_IN_S 秒缓启动（从 0 渐变到目标，M2 实测直发阶跃超调 ~72%）。
"""

from __future__ import annotations

import math
import random as _random

RAMP_IN_S = 2.0


def _ramp(t: float) -> float:
    return 1.0 if t >= RAMP_IN_S else (t / RAMP_IN_S)


def _seg_index(t: float, starts: list[float]) -> int:
    for i in range(len(starts) - 1):
        if t < starts[i + 1]:
            return i
    return len(starts) - 2


def make_traj(args):
    """根据 argparse args 生成 traj(t) -> (front_back, left_right)。"""
    if args.circle:
        amp, period = args.circle

        def traj(t):
            r = _ramp(t)
            return (amp * math.cos(2 * math.pi * t / period) * r,
                    amp * math.sin(2 * math.pi * t / period) * r)

    elif args.hold:
        fb0, lr0 = args.hold

        def traj(t):
            r = _ramp(t)
            return (fb0 * r, lr0 * r)

    elif args.lissajous:
        amp_fb, amp_lr, f1, f2 = args.lissajous

        def traj(t):
            r = _ramp(t)
            return (amp_fb * math.sin(2 * math.pi * f1 * t) * r,
                    amp_lr * math.sin(2 * math.pi * f2 * t) * r)

    elif args.eight:
        amp, period = args.eight
        f = 1.0 / period

        def traj(t):
            r = _ramp(t)
            return (amp * math.sin(2 * math.pi * f * t) * r,
                    amp * math.sin(4 * math.pi * f * t) * r)

    elif args.variable_circle:
        amp, period = args.variable_circle

        def traj(t):
            r = _ramp(t)
            # 变速：角速度带 25% 正弦调制（快慢交替）
            theta = 2 * math.pi * (t / period + 0.25 * math.sin(2 * math.pi * t / period))
            return (amp * math.cos(theta) * r,
                    amp * math.sin(theta) * r)

    elif getattr(args, "speed_ladder", None):
        # 速度阶梯（§8.4 D0）：同幅度依次跑多频率，覆盖 |q̇| 区间。
        # 两轴用【不同频率序列】（lr 反序），避免两轴长期锁相造成退化。
        amp = args.speed_ladder
        speeds = list(args.speeds)
        seg_dur = args.speed_seg_dur
        fade = 0.6  # 档内渐入/渐出，避免频率切换瞬态
        starts = [i * seg_dur for i in range(len(speeds) + 1)]

        def _ladder(t, seq):
            i = _seg_index(t, starts)
            tl = t - starts[i]
            env = min(1.0, tl / fade, max(0.0, (seg_dur - tl) / fade))
            return amp * math.sin(2 * math.pi * seq[i] * tl) * env

        seq_lr = list(reversed(speeds))

        def traj(t):
            r = _ramp(t)
            return (_ladder(t, speeds) * r, _ladder(t, seq_lr) * r)

    elif getattr(args, "random_fourier", None):
        # 随机多频 Fourier 叠加（§8.4 D0）：(q, q̇) 空间覆盖最大化。
        # 固定 seed 保证可复现；按 Σ|a| 归一化使峰值 ≤ amp。
        amp = args.random_fourier
        rng = _random.Random(args.seed)
        n_h = args.fourier_harmonics

        def _mk():
            hs = [(rng.uniform(0.03, args.fourier_fmax), rng.uniform(0.3, 1.0),
                   rng.uniform(0, 2 * math.pi)) for _ in range(n_h)]
            norm = sum(a for _, a, _ in hs)
            return [(f, a / norm, ph) for f, a, ph in hs]

        hs_fb = _mk()
        hs_lr = _mk()

        def traj(t):
            r = _ramp(t)
            fb = sum(a * math.sin(2 * math.pi * f * t + ph) for f, a, ph in hs_fb) * amp
            lr = sum(a * math.sin(2 * math.pi * f * t + ph) for f, a, ph in hs_lr) * amp
            return (fb * r, lr * r)

    elif args.waypoints:
        wps = _parse_waypoints(args.waypoints)
        # 累计段起始时间（段间 RAMP_IN_S 平滑过渡，之后驻留）
        starts = [0.0]
        for _, _, d in wps:
            starts.append(starts[-1] + d)

        def traj(t):
            for i, (fb, lr, _) in enumerate(wps):
                if t < starts[i + 1]:
                    t_local = t - starts[i]
                    if t_local < RAMP_IN_S:
                        prev = wps[i - 1] if i > 0 else (0.0, 0.0, 0.0)
                        r = t_local / RAMP_IN_S
                        return (prev[0] + (fb - prev[0]) * r,
                                prev[1] + (lr - prev[1]) * r)
                    return (fb, lr)
            return (wps[-1][0], wps[-1][1])

    else:

        def traj(t):
            return (0.0, 0.0)

    return traj


def _parse_waypoints(flat) -> list[tuple[float, float, float]]:
    """展平的 [fb, lr, dur, fb, lr, dur, ...] → [(fb, lr, dur), ...]。"""
    if len(flat) % 3 != 0:
        raise ValueError("--waypoints 参数需为 3 的倍数（fb lr dur 循环）")
    return [(float(flat[i]), float(flat[i + 1]), float(flat[i + 2]))
            for i in range(0, len(flat), 3)]
