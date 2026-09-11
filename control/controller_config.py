"""控制器参数配置（独立于标定，为后续前馈项留接口）。

职责拆分：
    - 标定 Calibration = 认识系统的结果：符号映射 + 基础增益（随硬件平台，相对固定）。
    - 控制器参数 ControllerConfig = 控制策略：前馈增益调度 + 各种前馈项（随优化迭代更新）。

前馈项接口（可扩展，新增前馈项时在此追加字段并在 load 里读取）：
    - direction_gains：方向分段增益（级 1，已实现）
    - gain_poly：参数化逆映射 g(q) 系数（级 1.5，已实现）
    - gain_cross：几何耦合解耦交叉项 c(q_other)（级 1.6，已实现，加性）
    - velocity_lead：速度前馈（相位超前）τ，u_ff = g(q_d + τ·q̇_d)（§8.4，D0 实证主导项）
    - dynamic_nn：学习型动态残差 MLP（§8.4 D1，已实现）——**残差式**：
      u_ff = g_static(q_d) + f_θ(q_d, q̇_d)；缺省/权重置零即精确退回纯稳态前馈。
    - hysteresis：迟滞补偿（级 2，预留）
    - friction：摩擦补偿（级 3，预留）
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ControllerConfig:
    direction_gains: dict | None = None   # {"fb": {"pos","neg"}, "lr": {"pos","neg"}}
    gain_poly: dict | None = None         # {"fb": [b0,b1,b2,b3], "lr": [...]} 逆映射 g(q_self)
    gain_cross: dict | None = None        # {"fb": [c1..], "lr": [c1..]} 交叉项 c(q_other)，无常数项
    velocity_lead: dict | None = None     # {"fb": tau_s, "lr": tau_s} 速度前馈相位超前（秒）
    dynamic_nn: dict | None = None        # 动态残差 MLP（§8.4 D1）：归一化参数 + 层权重
    slew_limit: float | None = None       # u_ff 每拍变化上限（offset/拍），None=不限
    hysteresis: dict | None = None        # {"fb": h, "lr": h} 迟滞补偿（offset），级 2
    # 未来扩展字段在此追加，load 时读取对应 key

    _last_uff: tuple = field(default=None, init=False, repr=False)  # 上次前馈输出（slew 用）

    def __post_init__(self):
        self._last_uff = None

    @classmethod
    def none(cls) -> "ControllerConfig":
        """无前馈（u_ff=0）。"""
        return cls(direction_gains=None)

    @classmethod
    def load(cls, path: str | Path) -> "ControllerConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            direction_gains=data.get("direction_gains"),
            gain_poly=data.get("gain_poly"),
            gain_cross=data.get("gain_cross"),
            velocity_lead=data.get("velocity_lead"),
            dynamic_nn=data.get("dynamic_nn"),
            slew_limit=data.get("slew_limit"),
            hysteresis=data.get("hysteresis"),
        )

    def has_feedforward(self) -> bool:
        """是否启用前馈。"""
        return (self.direction_gains is not None or self.gain_poly is not None
                or self.gain_cross is not None or self.dynamic_nn is not None
                or self.velocity_lead is not None)

    def feedforward(self, q_d_fb: float, q_d_lr: float,
                    qdot_d_fb: float = 0.0, qdot_d_lr: float = 0.0,
                    qddot_d_fb: float = 0.0, qddot_d_lr: float = 0.0) -> tuple[float, float]:
        """前馈反解：目标关节角（度）→ 差分 offset（含迟滞 + slew 整形）。

        静态基座 = own(q_self) + 交叉项 c(q_other)；own 依次回退
        gain_poly > direction_gains；级 2 = 迟滞项 h·sign(q̇_d)。
        """
        if self.gain_poly is not None:
            # 速度前馈（相位超前）：把静态逆映射按 τ·q̇_d 前瞻求值。
            # τ=0（或缺省）→ 与 g(q_d) **逐位相同**，零退回保证天然成立。
            q_eval_fb, q_eval_lr = q_d_fb, q_d_lr
            if self.velocity_lead is not None:
                q_eval_fb += self.velocity_lead.get("fb", 0.0) * qdot_d_fb
                q_eval_lr += self.velocity_lead.get("lr", 0.0) * qdot_d_lr
            u_fb = _poly(self.gain_poly["fb"], q_eval_fb)
            u_lr = _poly(self.gain_poly["lr"], q_eval_lr)
        elif self.direction_gains is not None:
            g = self.direction_gains
            u_fb = q_d_fb / (g["fb"]["pos"] if q_d_fb >= 0 else g["fb"]["neg"])
            u_lr = q_d_lr / (g["lr"]["pos"] if q_d_lr >= 0 else g["lr"]["neg"])
        else:
            u_fb = u_lr = 0.0

        # 级 1.6：几何耦合解耦交叉项（加性，无常数项）
        if self.gain_cross is not None:
            u_fb += _poly_no_const(self.gain_cross["fb"], q_d_lr)  # fb 输出受 lr 角影响
            u_lr += _poly_no_const(self.gain_cross["lr"], q_d_fb)  # lr 输出受 fb 角影响

        # §8.4 D1：学习型动态残差（**残差式**，只加修正量；缺省/零权重=退回稳态前馈）
        if self.dynamic_nn is not None:
            feats = [q_d_fb, q_d_lr, qdot_d_fb, qdot_d_lr]
            if int(self.dynamic_nn.get("n_in", 4)) >= 6:   # v2：含 q̈（T1/D0 实证有效）
                feats += [qddot_d_fb, qddot_d_lr]
            r_fb, r_lr = _nn_forward(self.dynamic_nn, feats)
            u_fb += r_fb
            u_lr += r_lr

        # 级 2：迟滞补偿 h·sign(q̇_d)
        if self.hysteresis is not None:
            u_fb += self.hysteresis.get("fb", 0.0) * (1.0 if qdot_d_fb >= 0 else -1.0)
            u_lr += self.hysteresis.get("lr", 0.0) * (1.0 if qdot_d_lr >= 0 else -1.0)

        # slew 速率整形：限制 u_ff 每拍变化量，防目标突变时前馈跳变
        if self.slew_limit is not None:
            if self._last_uff is not None:
                u_fb = _slew(self._last_uff[0], u_fb, self.slew_limit)
                u_lr = _slew(self._last_uff[1], u_lr, self.slew_limit)
        self._last_uff = (u_fb, u_lr)
        return u_fb, u_lr


def _poly(coeffs, x):
    """多项式求值 u = b0 + b1·x + b2·x² + ..."""
    return sum(c * x ** k for k, c in enumerate(coeffs))


def _poly_no_const(coeffs, x) -> float:
    """无常数项多项式求值 u = c1·x + c2·x² + ...（coeffs[0] 对应 x¹）。"""
    return sum(c * x ** (k + 1) for k, c in enumerate(coeffs))


def _nn_forward(nn: dict, features: list[float]) -> tuple[float, float]:
    """动态残差 MLP 前向（纯 Python，运行时无 torch 依赖）。

    nn 结构：{"in_mean","in_std","out_mean","out_std","layers":[{"W","b"}...],"act","clip"}
    输入先标准化 → 逐层 (act 除末层) → 输出去标准化 → 按 clip 限幅（防外推发散）。
    """
    x = [(f - m) / (s if s else 1.0) for f, m, s in zip(features, nn["in_mean"], nn["in_std"])]
    layers = nn["layers"]
    act = nn.get("act", "tanh")
    for i, layer in enumerate(layers):
        W, b = layer["W"], layer["b"]
        x = [sum(w * xi for w, xi in zip(row, x)) + bi for row, bi in zip(W, b)]
        if i < len(layers) - 1:
            if act == "tanh":
                x = [math.tanh(v) for v in x]
            elif act == "relu":
                x = [v if v > 0.0 else 0.0 for v in x]
    out = [v * s + m for v, m, s in zip(x, nn["out_mean"], nn["out_std"])]
    clip = nn.get("clip")
    if clip is not None:
        out = [max(-clip, min(clip, v)) for v in out]
    return out[0], out[1]


def _slew(prev: float, cur: float, limit: float) -> float:
    """斜率限制：每拍变化量不超过 limit。"""
    d = cur - prev
    if abs(d) > limit:
        return prev + limit * (1 if d > 0 else -1)
    return cur
