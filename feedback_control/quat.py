"""四元数 → (pitch, yaw) 分解（纯函数，无外部依赖）。

约定：相对旋转 R = Ry(yaw) @ Rx(pitch)，与 geometric_ik 的 IK 一致。
    pitch = atan2(-R[1,2], R[1,1])
    yaw   = atan2( R[0,2], R[2,2])
其中 R 由相对四元数 q_rel = inv(ref) * q 得到。
"""

from __future__ import annotations

import math


def quat_to_pitch_yaw(
    qw: float, qx: float, qy: float, qz: float,
    ref=(1.0, 0.0, 0.0, 0.0),
) -> tuple[float, float]:
    """四元数 (w,x,y,z) → (pitch_deg, yaw_deg)，相对参考四元数 ref。

    ref 为中立位参考四元数，默认单位四元数（即绝对姿态）。
    """
    rw, rx, ry, rz = ref
    n = rw * rw + rx * rx + ry * ry + rz * rz

    # q_rel = inv(ref) * q
    w = (rw * qw + rx * qx + ry * qy + rz * qz) / n
    x = (rw * qx - rx * qw - ry * qz + rz * qy) / n
    y = (rw * qy + rx * qz - ry * qw - rz * qx) / n
    z = (rw * qz - rx * qy + ry * qx - rz * qw) / n

    r11 = 1 - 2 * (x * x + z * z)   # R[1,1]
    r12 = 2 * (y * z - x * w)       # R[1,2]
    r02 = 2 * (x * z + y * w)       # R[0,2]
    r22 = 1 - 2 * (x * x + y * y)   # R[2,2]

    pitch = math.degrees(math.atan2(-r12, r11))
    yaw = math.degrees(math.atan2(r02, r22))
    return pitch, yaw
