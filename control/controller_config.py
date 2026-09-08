"""控制器参数配置（独立于标定，为后续前馈项留接口）。

职责拆分：
    - 标定 Calibration = 认识系统的结果：符号映射 + 基础增益（随硬件平台，相对固定）。
    - 控制器参数 ControllerConfig = 控制策略：前馈增益调度 + 各种前馈项（随优化迭代更新）。

前馈项接口（可扩展，新增前馈项时在此追加字段并在 load 里读取）：
    - direction_gains：方向分段增益（级 1，已实现）
    - amplitude_gains：幅值增益调度（级 1.5，预留）
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
    # 未来扩展字段在此追加，load 时读取对应 key

    @classmethod
    def none(cls) -> "ControllerConfig":
        """无前馈（u_ff=0）。"""
        return cls(direction_gains=None)

    @classmethod
    def load(cls, path: str | Path) -> "ControllerConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            direction_gains=data.get("direction_gains"),
        )

    def has_feedforward(self) -> bool:
        """是否启用前馈。"""
        return self.direction_gains is not None

    def feedforward(self, q_d_fb: float, q_d_lr: float) -> tuple[float, float]:
        """前馈反解：目标关节角（度）→ 差分 offset。

        无 direction_gains 时返回 (0,0)（等价 u_ff=0）。
        """
        if self.direction_gains is None:
            return 0.0, 0.0
        g = self.direction_gains
        u_fb = q_d_fb / (g["fb"]["pos"] if q_d_fb >= 0 else g["fb"]["neg"])
        u_lr = q_d_lr / (g["lr"]["pos"] if q_d_lr >= 0 else g["lr"]["neg"])
        return u_fb, u_lr
