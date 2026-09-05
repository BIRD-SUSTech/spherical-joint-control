"""轨迹定义：球关节姿态序列。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Waypoint:
    """单个轨迹点。

    Attributes:
        pitch_deg: 目标 pitch (绕 X 轴, 度).
        yaw_deg: 目标 yaw (绕 Y 轴, 度).
        duration_s: 在该点停留的时间 (s).
    """
    pitch_deg: float
    yaw_deg: float
    duration_s: float


@dataclass
class Trajectory:
    """姿态序列轨迹。"""
    waypoints: List[Waypoint] = field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return sum(wp.duration_s for wp in self.waypoints)

    @classmethod
    def from_pairs(
        cls,
        pairs: list[tuple[float, float, float]],
    ) -> "Trajectory":
        """从 (pitch_deg, yaw_deg, duration_s) 元组列表构造。

        Example:
            traj = Trajectory.from_pairs([
                (0, 0, 3),     # 中立位停留 3s
                (10, 0, 5),    # pitch=10 停留 5s
                (0, 10, 5),    # yaw=10 停留 5s
                (0, 0, 3),     # 回到中立
            ])
        """
        return cls([Waypoint(p, y, d) for p, y, d in pairs])

    def __len__(self) -> int:
        return len(self.waypoints)

    def __iter__(self):
        return iter(self.waypoints)


# ---- 轨迹生成器 ----

def make_circle_trajectory(
    radius_deg: float,
    period_s: float,
    steps: int = 36,
) -> Trajectory:
    """生成圆形轨迹 (pitch-yaw 平面内画圆)。

    Args:
        radius_deg: 圆半径 (度).
        period_s: 完整一圈的时间 (s).
        steps: 轨迹点数.
    """
    dt = period_s / steps
    waypoints = []
    for i in range(steps + 1):
        phase = 2 * math.pi * i / steps
        pitch = radius_deg * math.cos(phase)
        yaw = radius_deg * math.sin(phase)
        dur = dt if i < steps else 0.0  # 最后一个点不停留
        if dur > 0:
            waypoints.append(Waypoint(pitch, yaw, dur))
    return Trajectory(waypoints)


def make_sine_trajectory(
    pitch_amplitude_deg: float,
    yaw_amplitude_deg: float,
    period_s: float,
    steps: int = 36,
) -> Trajectory:
    """生成正交正弦轨迹 (pitch 和 yaw 相位差 90°)。

    Args:
        pitch_amplitude_deg: pitch 方向振幅.
        yaw_amplitude_deg: yaw 方向振幅.
        period_s: 周期 (s).
        steps: 轨迹点数.
    """
    dt = period_s / steps
    waypoints = []
    for i in range(steps + 1):
        phase = 2 * math.pi * i / steps
        pitch = pitch_amplitude_deg * math.sin(phase)
        yaw = yaw_amplitude_deg * math.sin(phase + math.pi / 2)
        dur = dt if i < steps else 0.0
        if dur > 0:
            waypoints.append(Waypoint(pitch, yaw, dur))
    return Trajectory(waypoints)


def make_hold_trajectory(pitch_deg: float, yaw_deg: float, duration_s: float) -> Trajectory:
    """生成静止保持轨迹 (单点)."""
    return Trajectory([Waypoint(pitch_deg, yaw_deg, duration_s)])
