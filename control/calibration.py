"""标定配置加载与姿态映射（替代 FLIP 硬编码，设计文档 §7.3 收口）。

默认标定 = M1/M2 实机初步确认的映射（前后←roll 正号，左右←pitch 正号）。
标定后加载 JSON（control.calibrate 产出）覆盖默认。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Calibration:
    front_back_euler: str = "roll"      # "roll" | "pitch"
    left_right_euler: str = "pitch"     # "roll" | "pitch"
    front_back_sign: int = 1            # +1 | -1
    left_right_sign: int = 1            # +1 | -1
    gain_deg_per_offset: dict = None    # {"front_back": g, "left_right": g}

    def __post_init__(self):
        if self.gain_deg_per_offset is None:
            # 缺省 = rig1 的 M4 实测 ±5° 有效增益（单一事实源，见 calibrations/rig1.json）
            self.gain_deg_per_offset = {"front_back": 0.057, "left_right": 0.059}

    @classmethod
    def default(cls) -> "Calibration":
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
        )

    def map_pose(self, roll: float, pitch: float) -> tuple[float, float]:
        """动捕原始欧拉角 → (前后, 左右) 关节角（度），含符号。"""
        fb = (roll if self.front_back_euler == "roll" else pitch) * self.front_back_sign
        lr = (pitch if self.left_right_euler == "pitch" else roll) * self.left_right_sign
        return fb, lr
