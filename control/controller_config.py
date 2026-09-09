"""控制器参数配置（独立于标定，为后续前馈项留接口）。

职责拆分：
    - 标定 Calibration = 认识系统的结果：符号映射 + 基础增益（随硬件平台，相对固定）。
    - 控制器参数 ControllerConfig = 控制策略：前馈增益调度 + 各种前馈项（随优化迭代更新）。

前馈项接口（可扩展，新增前馈项时在此追加字段并在 load 里读取）：
    - direction_gains：方向分段增益（级 1，已实现）
    - gain_poly：参数化逆映射 g(q) 系数（级 1.5，已实现）
    - gain_cross：几何耦合解耦交叉项 c(q_other)（级 1.6，已实现，加性）
    - hysteresis：迟滞补偿（级 2，预留）
    - friction：摩擦补偿（级 3，预留）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ControllerConfig:
    direction_gains: dict | None = None   # {"fb": {"pos","neg"}, "lr": {"pos","neg"}}
    gain_poly: dict | None = None         # {"fb": [b0,b1,b2,b3], "lr": [...]} 逆映射 g(q_self)
    gain_cross: dict | None = None        # {"fb": [c1..], "lr": [c1..]} 交叉项 c(q_other)，无常数项
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
            slew_limit=data.get("slew_limit"),
            hysteresis=data.get("hysteresis"),
        )

    def has_feedforward(self) -> bool:
        """是否启用前馈。"""
        return (self.direction_gains is not None or self.gain_poly is not None
                or self.gain_cross is not None)

    def feedforward(self, q_d_fb: float, q_d_lr: float,
                    qdot_d_fb: float = 0.0, qdot_d_lr: float = 0.0) -> tuple[float, float]:
        """前馈反解：目标关节角（度）→ 差分 offset（含迟滞 + slew 整形）。

        静态基座 = own(q_self) + 交叉项 c(q_other)；own 依次回退
        gain_poly > direction_gains；级 2 = 迟滞项 h·sign(q̇_d)。
        """
        if self.gain_poly is not None:
            u_fb = _poly(self.gain_poly["fb"], q_d_fb)
            u_lr = _poly(self.gain_poly["lr"], q_d_lr)
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


def _slew(prev: float, cur: float, limit: float) -> float:
    """斜率限制：每拍变化量不超过 limit。"""
    d = cur - prev
    if abs(d) > limit:
        return prev + limit * (1 if d > 0 else -1)
    return cur
