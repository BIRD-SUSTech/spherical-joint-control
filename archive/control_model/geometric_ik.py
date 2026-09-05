"""球关节几何逆运动学模型。

两自由度定义为 pitch (绕 X 轴) 和 yaw (绕 Y 轴)。

基于 MATLAB example_code/kinematics 中的几何参数：
  - 球体半径 R = 25mm
  - 4 条缆绳紧贴球表面，沿大圆弧路径
  - 固定点位于球顶部圆环 (phi=41°)
  - 球面点位于球底部圆环 (phi=137°)
  - 四点绕 Z 轴均匀分布，间距 90°

输出缆绳长度变化量（相对于球杆竖直中立状态）。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---- 几何参数 ----
R: float = 25.0  # 球体半径 (mm)

# 固定点球坐标 [phi_deg, theta_deg]
PHI_FIX_DEG: float = 41.0    # 顶部圆环
PHI_BALL_DEG: float = 137.0  # 底部圆环

# 四点绕 Z 轴方位角 (度) — 对应物理舵机 1(下) 2(右) 3(上) 4(左)
THETA_OFFSETS_DEG: tuple[float, float, float, float] = (90.0, 180.0, 270.0, 0.0)


def _sph2cart(phi_deg: float, theta_deg: float) -> NDArray:
    """球坐标 → 直角坐标 (phi: 与 -Z 轴夹角, theta: XY 平面方位角)."""
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    return R * np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        -np.cos(phi),
    ])


def _rx(pitch_deg: float) -> NDArray:
    """绕 X 轴旋转矩阵 (pitch)."""
    a = np.deg2rad(pitch_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(yaw_deg: float) -> NDArray:
    """绕 Y 轴旋转矩阵 (yaw)."""
    b = np.deg2rad(yaw_deg)
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _great_circle_arc(p1: NDArray, p2: NDArray) -> float:
    """球面上两点间的大圆弧长度 (mm)."""
    u1 = p1 / np.linalg.norm(p1)
    u2 = p2 / np.linalg.norm(p2)
    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(R * np.arccos(dot))


class GeometricIK:
    """几何逆运动学：给定球杆姿态角，计算四条缆绳相对于中立位的长度变化."""

    def __init__(self):
        self._P_fix: NDArray = np.array([
            _sph2cart(PHI_FIX_DEG, th) for th in THETA_OFFSETS_DEG
        ])

        self._P_ball_neutral: NDArray = np.array([
            _sph2cart(PHI_BALL_DEG, th) for th in THETA_OFFSETS_DEG
        ])

        self._L_neutral: NDArray = np.array([
            _great_circle_arc(self._P_fix[i], self._P_ball_neutral[i])
            for i in range(4)
        ])

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def neutral_lengths(self) -> NDArray:
        """中立位四条缆绳弧长 (mm)."""
        return self._L_neutral.copy()

    @property
    def fix_points(self) -> NDArray:
        """固定点直角坐标 (4, 3)."""
        return self._P_fix.copy()

    @property
    def ball_points_neutral(self) -> NDArray:
        """中立位球面点直角坐标 (4, 3)."""
        return self._P_ball_neutral.copy()

    def solve(self, pitch_deg: float, yaw_deg: float) -> NDArray:
        """给定姿态角，返回四条缆绳相对于中立位的长度变化量 (mm).

        Args:
            pitch_deg: 绕 X 轴转角 (pitch, 度).
            yaw_deg: 绕 Y 轴转角 (yaw, 度).

        Returns:
            delta_lengths: shape (4,) — 正值表示缆绳需从中立位放长.

        Raises:
            ValueError: 若角度超出 ±60° (结构限制).
        """
        if abs(pitch_deg) > 60.0 or abs(yaw_deg) > 60.0:
            raise ValueError(
                f"角度超出结构限位 ±60°: pitch={pitch_deg}, yaw={yaw_deg}"
            )

        R_total = _ry(yaw_deg) @ _rx(pitch_deg)

        P_ball_rotated = np.array([
            R_total @ self._P_ball_neutral[i] for i in range(4)
        ])

        L_current = np.array([
            _great_circle_arc(self._P_fix[i], P_ball_rotated[i])
            for i in range(4)
        ])

        return L_current - self._L_neutral

    def solve_from_rotation(self, R_mat: NDArray) -> NDArray:
        """给定 3x3 旋转矩阵，返回缆绳长度变化量.

        Args:
            R_mat: (3, 3) 旋转矩阵.

        Returns:
            delta_lengths: shape (4,).
        """
        R_mat = np.asarray(R_mat)
        if R_mat.shape != (3, 3):
            raise ValueError(f"旋转矩阵必须为 (3,3), 实际: {R_mat.shape}")

        P_ball_rotated = np.array([
            R_mat @ self._P_ball_neutral[i] for i in range(4)
        ])

        L_current = np.array([
            _great_circle_arc(self._P_fix[i], P_ball_rotated[i])
            for i in range(4)
        ])

        return L_current - self._L_neutral

    def cable_lengths(self, pitch_deg: float, yaw_deg: float) -> NDArray:
        """给定姿态角，返回绝对缆绳弧长 (mm)，不含差值计算."""
        R_total = _ry(yaw_deg) @ _rx(pitch_deg)
        P_ball_rotated = np.array([
            R_total @ self._P_ball_neutral[i] for i in range(4)
        ])
        return np.array([
            _great_circle_arc(self._P_fix[i], P_ball_rotated[i])
            for i in range(4)
        ])

    def jacobian(self, pitch_deg: float, yaw_deg: float, h: float = 0.1) -> NDArray:
        """数值中心差分 Jacobian: d(ΔL) / d(pitch, yaw)，shape (4, 2).

        Args:
            pitch_deg: 当前 pitch (度).
            yaw_deg: 当前 yaw (度).
            h: 差分步长 (度).

        Returns:
            J[i, 0] = ∂ΔL_i / ∂pitch, J[i, 1] = ∂ΔL_i / ∂yaw (mm/度).

        Raises:
            ValueError: 若差分点超出 ±60° 结构限位.
        """
        J = np.zeros((4, 2))
        Lp = self.solve(pitch_deg + h, yaw_deg)
        Lm = self.solve(pitch_deg - h, yaw_deg)
        J[:, 0] = (Lp - Lm) / (2 * h)
        Ly = self.solve(pitch_deg, yaw_deg + h)
        Ln = self.solve(pitch_deg, yaw_deg - h)
        J[:, 1] = (Ly - Ln) / (2 * h)
        return J
