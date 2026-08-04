"""基于逆运动学的开环控制器。

控制流程：
    target pitch/yaw ──→ GeometricIK.solve() ──→ ΔL[4] (mm) ──→ 归一化舵机 [-1, 1]

无反馈，纯前馈。适用于标定数据采集等需要固定轨迹的场景。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..geometric_ik import GeometricIK


# ---- 舵机参数 ----
DRUM_RADIUS_MM: float = 18.0     # 舵盘半径 (mm)
SERVO_TOTAL_DEG: float = 270.0    # 舵机总行程 (度)
SERVO_HALF_DEG: float = 135.0     # 半行程, 归一化基准


def _cable_delta_to_servo_norm(delta_L_mm: NDArray, drum_radius_mm: float) -> NDArray:
    """缆长变化量 → 归一化舵机角度 [-1, 1]."""
    delta_theta_deg = delta_L_mm / drum_radius_mm * (180.0 / np.pi)
    return delta_theta_deg / SERVO_HALF_DEG


class OpenLoopController:
    """开环控制器：目标姿态直接经 IK 转换为舵机指令，无反馈修正。

    用法:
        ctrl = OpenLoopController()
        servo_cmd = ctrl.command(pitch_deg=15.0, yaw_deg=-10.0)
    """

    def __init__(self, drum_radius_mm: float | None = None):
        self._ik = GeometricIK()
        self._drum_radius = drum_radius_mm or DRUM_RADIUS_MM

    @property
    def ik(self) -> GeometricIK:
        return self._ik

    @property
    def drum_radius_mm(self) -> float:
        return self._drum_radius

    def command(self, pitch_deg: float, yaw_deg: float) -> NDArray:
        """给定目标姿态，返回归一化舵机角度。

        Args:
            pitch_deg: 目标 pitch (绕 X 轴, 度).
            yaw_deg: 目标 yaw (绕 Y 轴, 度).

        Returns:
            servo_norm: shape (4,), 归一化舵机角度 [-1, 1], 0 = 中位.

        Raises:
            ValueError: 若角度超出 ±60°.
        """
        delta_L = self._ik.solve(pitch_deg, yaw_deg)
        return _cable_delta_to_servo_norm(delta_L, self._drum_radius)
