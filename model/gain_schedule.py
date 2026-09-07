"""增益调度前馈（§8.2 级 1 最简形）：方向分段增益 G(sign(u))。

用开环数据 (u, q) 拟合每轴正负方向的增益（过原点最小二乘），
前馈反解 u_ff = q_d / g(sign(q_d))。纯静态基座，数据现成（M4 开环），不需闭环收敛段。

用途：作为 g 的线性基座（u_linear），先让闭环能在大范围稳定，
为 g 机制 A 采集闭环收敛段数据铺路（§13 螺旋第一轮）。
"""

from __future__ import annotations

import numpy as np


def _slope(u: np.ndarray, q: np.ndarray) -> float:
    """过原点最小二乘斜率 g = Σ(q·u)/Σ(u²)。"""
    if len(u) == 0 or float((u ** 2).sum()) == 0.0:
        return 0.0
    return float((q * u).sum() / (u ** 2).sum())


def fit_direction_gains(u: np.ndarray, q: np.ndarray) -> dict:
    """每轴正负方向各一个增益。返回 {"fb": {"pos": g, "neg": g}, "lr": {...}}。"""
    gains = {}
    for a, name in enumerate(["fb", "lr"]):
        ua, qa = u[:, a], q[:, a]
        gains[name] = {
            "pos": _slope(ua[ua > 0], qa[ua > 0]),
            "neg": _slope(ua[ua < 0], qa[ua < 0]),
        }
    return gains


def feedforward(gains: dict, q_d: np.ndarray) -> np.ndarray:
    """前馈反解：u_ff = q_d / g(sign(q_d))。q_d 形状 (2,) → u_ff (2,)。"""
    u_ff = np.zeros(2)
    for a, name in enumerate(["fb", "lr"]):
        d = q_d[a]
        g = gains[name]["pos"] if d >= 0 else gains[name]["neg"]
        if abs(g) > 1e-9:
            u_ff[a] = d / g
    return u_ff
