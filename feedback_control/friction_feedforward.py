"""结构化摩擦前馈：d = G·Δq + f_c·sign(Δq) + f_s·sign(Δq)·起步脉冲。

加载 friction_config.json（由 feedforward.fit_friction 生成）。
死区内（|Δq| < deadband）不输出，交给 PID 保持。
"""

from __future__ import annotations

import json

import numpy as np


def decode(d, p):
    d1, d2 = float(d[0]), float(d[1])
    n = np.array([p + d1 / 2.0, p + d2 / 2.0, p - d1 / 2.0, p - d2 / 2.0])
    return np.clip(n, -1.0, 1.0)


class FrictionFeedforward:
    def __init__(self, config_path):
        cfg = json.load(open(config_path))
        self.p = float(cfg["pretension"])
        self.G = np.array([cfg["pitch_G"], cfg["yaw_G"]])
        self.f_c = np.array([cfg["pitch_f_c"], cfg["yaw_f_c"]])
        self.f_s = np.array([cfg["pitch_f_s"], cfg["yaw_f_s"]])
        self.deadband = np.array([cfg["pitch_deadband_deg"], cfg["yaw_deadband_deg"]])
        self._prev_sign = np.zeros(2)

    def compute(self, dq_d):
        """dq_d: (2,) 期望增量(°/拍) -> 差分 d (2,) norm。"""
        dq = np.asarray(dq_d, dtype=float)
        d = np.zeros(2)
        for i in range(2):
            if abs(dq[i]) < self.deadband[i]:
                d[i] = 0.0
                self._prev_sign[i] = 0
            else:
                s = float(np.sign(dq[i]))
                d[i] = self.G[i] * dq[i] + self.f_c[i] * s
                if s != self._prev_sign[i]:        # 方向刚翻转 -> 静摩擦起步脉冲
                    d[i] += self.f_s[i] * s
                    self._prev_sign[i] = s
        return d

    def u_ff(self, dq_d):
        return decode(self.compute(dq_d), self.p)
