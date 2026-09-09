"""标定配置加载与姿态映射（替代 FLIP 硬编码，设计文档 §7.3 收口）。

职责：仅"认识系统"的结果——符号映射 + 基础增益（随硬件平台，相对固定）。
前馈增益调度等控制器参数在 `control/controller_config.py`（独立配置，可迭代）。

未来字段（设计文档 §8.3.2）：
    - co_tension：共模预紧 c_fb(q_lr)/c_lr(q_fb)（每对两缆同收的偶函数），属标定而非
      控制器参数；依赖固件共模通道（当前无），需在 M4 开环辨识之前标定。

默认标定 = M1/M2 实机初步确认的映射（前后←roll 正号，左右←pitch 正号）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# rig1（旧硬件）遗留默认增益：仅作 M1/M2 最小闭环的无标定占位（闭环 PID 不消费增益）。
# 新硬件（rig2）舵机更强、真实增益更大，open-loop 激励【禁止】使用此默认值——
# 开环前必须先跑 M3（control/calibrate.py）测得 calibrations/rig2.json 的真实增益。
_LEGACY_GAINS = {"front_back": 0.057, "left_right": 0.059}


@dataclass
class Calibration:
    front_back_euler: str = "roll"      # "roll" | "pitch"
    left_right_euler: str = "pitch"     # "roll" | "pitch"
    front_back_sign: int = 1            # +1 | -1
    left_right_sign: int = 1            # +1 | -1
    gain_deg_per_offset: dict = field(default_factory=lambda: dict(_LEGACY_GAINS))
                                        # {"front_back": g, "left_right": g}；None=未标定
    co_tension: dict = field(default_factory=dict)
                                        # {"front_back": [c0,c2,...], "left_right": [c0,c2,...]}
                                        # 共模预紧偶多项式 c(q_orth)=c0+c2·q²+…（§8.3.2）

    @classmethod
    def default(cls) -> "Calibration":
        """无标定占位（仅 M1/M2 闭环用，符号为结构先验；增益是旧值，开环不可用）。"""
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            front_back_euler=data.get("front_back_euler", "roll"),
            left_right_euler=data.get("left_right_euler", "pitch"),
            front_back_sign=data.get("front_back_sign", 1),
            left_right_sign=data.get("left_right_sign", 1),
            gain_deg_per_offset=data.get("gain_deg_per_offset"),
            co_tension=data.get("co_tension") or {},
        )

    def map_pose(self, roll: float, pitch: float) -> tuple[float, float]:
        """动捕原始欧拉角 → (前后, 左右) 关节角（度），含符号。"""
        fb = (roll if self.front_back_euler == "roll" else pitch) * self.front_back_sign
        lr = (pitch if self.left_right_euler == "pitch" else roll) * self.left_right_sign
        return fb, lr

    def has_co_tension(self) -> bool:
        """是否启用共模预紧（标定含 co_tension 且固件支持 id=3/4）。"""
        return bool(self.co_tension.get("front_back") or self.co_tension.get("left_right"))

    def common_mode(self, q_fb: float, q_lr: float) -> tuple[int, int]:
        """共模预紧 offset：c_fb(q_lr) 收紧前后对、c_lr(q_fb) 收紧左右对（§8.3.2）。

        偶多项式 c(q_orth) = c0 + c2·q_orth² + …；用另一轴【实测】角（松弛是实际位姿的
        函数）。仅当 has_co_tension() 为真才调用，否则返回 (0,0)。
        """
        if not self.has_co_tension():
            return 0, 0
        c_fb = _even_poly(self.co_tension.get("front_back", []), q_lr)
        c_lr = _even_poly(self.co_tension.get("left_right", []), q_fb)
        return int(round(c_fb)), int(round(c_lr))


def _even_poly(coeffs, q) -> float:
    """偶多项式求值 c0 + c2·q² + c4·q⁴ + …（coeffs[0] 对应 q⁰，coeffs[1] 对应 q²）。"""
    return sum(c * q ** (2 * k) for k, c in enumerate(coeffs))
