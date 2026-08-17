"""舵机指令编码（当前固件协议）。

当前固件直接通过串口接收 4 个舵机角度值（逗号分隔、单位 °、半行程 ±135、以 \\n 结尾），
固件本身不做对侧耦合。因此“对侧绳同步”在主机侧实现：

    2 个 DOF 控制量 (u_pitch, u_yaw)
        → 4 个舵机角度 [s1, s2, s3, s4]
        s1 = +u_pitch, s3 = -u_pitch   （pitch 对抗对 1↔3）
        s2 = +u_yaw,   s4 = -u_yaw     （yaw   对抗对 2↔4）

这与参考固件 Coupled_Servo(CH1,CH3,…) / Coupled_Servo(CH2,CH4,…) 的
“一端 +offset、另一端 -offset”语义等价，只是耦合位置从固件移到了主机。

方向约定：若某轴运动方向相反，把对应 flip 置 True（或给 PID 目标取负）。
"""

from __future__ import annotations

HALF_RANGE_DEG = 135


def coupled_angles(
    u_pitch: float,
    u_yaw: float,
    flip_pitch: bool = False,
    flip_yaw: bool = False,
) -> list[int]:
    """2 DOF 控制量 → 4 舵机整数角度 [s1, s2, s3, s4]，并限幅到 ±135。"""
    if flip_pitch:
        u_pitch = -u_pitch
    if flip_yaw:
        u_yaw = -u_yaw
    raw = [u_pitch, u_yaw, -u_pitch, -u_yaw]
    return [int(round(max(-HALF_RANGE_DEG, min(HALF_RANGE_DEG, a)))) for a in raw]


def format_command(angles) -> bytes:
    """4 个整数角度 → 当前固件协议帧（逗号分隔 + 换行）。"""
    return (",".join(str(int(a)) for a in angles) + "\n").encode("ascii")
