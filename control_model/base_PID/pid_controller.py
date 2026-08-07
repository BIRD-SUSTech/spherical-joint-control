"""IK 前馈 + PID 反馈控制器。

控制目标：球杆系统 pitch (绕 X 轴) 和 yaw (绕 Y 轴) 姿态。
控制量：4 路舵机角度，归一化到 [-1, 1]，0 为中位 (135°/270° 中位)。

数据流：
    target pose ──→ IK 前馈 ──→ ΔL_ff ──┐
                                          ├──→ 舵机归一化角度 [-1, 1]
    姿态误差 ──→ PID 反馈 ──→ ΔL_fb ──┘

PID 输出被解释为对 target pose 的修正 (再经 IK 转换为缆长变化)，
避免直接求解 Jacobian。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..geometric_ik import GeometricIK


@dataclass
class PIDGains:
    """PID 增益。"""
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    integral_max: float = 10.0        # 积分项限幅 (deg)
    output_max: float = 20.0           # PID 总输出限幅 (deg)


class PID:
    """离散 PID 控制器，带积分抗饱和。"""

    def __init__(self, gains: PIDGains | None = None):
        self._g = gains or PIDGains()
        self._integral: float = 0.0
        self._prev_error: float | None = None

    def update(self, error: float, dt: float) -> float:
        """计算控制量。

        Args:
            error: 当前误差 (目标 − 实际).
            dt: 时间步长 (s).

        Returns:
            控制量 (与 error 同单位).
        """
        g = self._g

        # 比例项
        P = g.Kp * error

        # 积分项 (梯形积分，带限幅)
        self._integral += g.Ki * error * dt
        self._integral = np.clip(self._integral, -g.integral_max, g.integral_max)
        I = self._integral

        # 微分项 (对测量微分，避免微分冲击)
        D = 0.0
        if self._prev_error is not None and dt > 1e-9:
            D = g.Kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = P + I + D
        return float(np.clip(output, -g.output_max, g.output_max))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None


class BasePIDController:
    """IK 前馈 + PID 反馈球关节控制器。

    两种工作模式（use_ik）：
    - True (默认): PID 输出修正目标姿态，经 GeometricIK 转缆长再转舵机归一化角度。
    - False: 不依赖 IK 模型，PID 输出直接映射为对抗对差分
      (pitch→servo_0/2, yaw→servo_1/3)，适用于 IK 模型与实际运动学偏差较大时。

    用法:
        ctrl = BasePIDController()
        servo_cmd = ctrl.update(
            target_pitch=10.0, target_yaw=0.0,
            current_pitch=9.5, current_yaw=-0.3,
            dt=0.01,
        )
    """

    # ---- 舵机参数 ----
    DRUM_RADIUS_MM: float = 18.0   # 舵盘半径 (mm)
    SERVO_TOTAL_DEG: float = 270.0  # 舵机总行程 (度)
    SERVO_MID_DEG: float = 135.0    # 舵机中位角度 (度)
    SERVO_HALF_DEG: float = 135.0   # 半行程 (= SERVO_TOTAL_DEG / 2)

    def __init__(
        self,
        pitch_pid: PID | None = None,
        yaw_pid: PID | None = None,
        drum_radius_mm: float | None = None,
        pretension_mm: float = 0.0,
        filter_tau_s: float = 0.05,
        use_ik: bool = True,
        direct_gain: float = 0.01,
        direct_pretension_norm: float = 0.0,
    ):
        self._ik = GeometricIK()
        self._pid_pitch = pitch_pid or PID(PIDGains(Kp=1.0, Ki=0.3, integral_max=15.0))
        self._pid_yaw = yaw_pid or PID(PIDGains(Kp=1.0, Ki=0.3, integral_max=15.0))
        self._drum_radius = drum_radius_mm or self.DRUM_RADIUS_MM
        self._pretension_mm = pretension_mm
        self._filter_tau_s = filter_tau_s
        self._use_ik = use_ik
        self._direct_gain = direct_gain
        self._direct_pretension_norm = direct_pretension_norm
        self._filtered_pitch: float | None = None
        self._filtered_yaw: float | None = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def ik(self) -> GeometricIK:
        return self._ik

    @property
    def pitch_pid(self) -> PID:
        return self._pid_pitch

    @property
    def yaw_pid(self) -> PID:
        return self._pid_yaw

    def update(
        self,
        target_pitch_deg: float,
        target_yaw_deg: float,
        current_pitch_deg: float,
        current_yaw_deg: float,
        dt: float,
    ) -> NDArray:
        """单步控制更新。

        Args:
            target_pitch_deg: 目标 pitch (绕 X 轴, 度).
            target_yaw_deg: 目标 yaw (绕 Y 轴, 度).
            current_pitch_deg: 当前测量 pitch (度).
            current_yaw_deg: 当前测量 yaw (度).
            dt: 距上次调用的时间间隔 (s).

        Returns:
            servo_norm: shape (4,), 归一化舵机角度 [-1, 1].
        """
        # EMA 低通滤波 — 平滑动捕噪声
        current_pitch_deg, current_yaw_deg = self._filter_pose(
            current_pitch_deg, current_yaw_deg, dt,
        )

        # PID 反馈 → 姿态修正量
        e_pitch = target_pitch_deg - current_pitch_deg
        e_yaw = target_yaw_deg - current_yaw_deg

        pitch_fb = self._pid_pitch.update(e_pitch, dt)
        yaw_fb = self._pid_yaw.update(e_yaw, dt)

        if self._use_ik:
            # 前馈 + 反馈合成目标姿态
            cmd_pitch = target_pitch_deg + pitch_fb
            cmd_yaw = target_yaw_deg + yaw_fb

            # IK → 缆长变化量
            delta_L = self._ik.solve(cmd_pitch, cmd_yaw)

            # 缆长 → 归一化舵机角度
            return self._cable_delta_to_servo_norm(delta_L)

        # 无 IK：PID 输出直接映射为对抗对差分。
        # Pitch 对 = servo_0/2, Yaw 对 = servo_1/3，符号与中立位 IK 一致。
        k = self._direct_gain
        bias = self._direct_pretension_norm
        norm = np.array([
            -k * pitch_fb + bias,   # servo 0
            -k * yaw_fb + bias,     # servo 1
            +k * pitch_fb + bias,   # servo 2 (与 0 对抗)
            +k * yaw_fb + bias,     # servo 3 (与 1 对抗)
        ])
        return np.clip(norm, -1.0, 1.0)

    def reset(self) -> None:
        """重置 PID 状态及滤波器。"""
        self._pid_pitch.reset()
        self._pid_yaw.reset()
        self._filtered_pitch = None
        self._filtered_yaw = None

    def _filter_pose(
        self, pitch: float, yaw: float, dt: float,
    ) -> tuple[float, float]:
        """指数移动平均低通滤波。

        tau ≤ 0 时直接返回原始值；首帧直接采纳，后续按 alpha = dt/(tau+dt) 平滑。
        """
        if self._filter_tau_s <= 0:
            return pitch, yaw
        if self._filtered_pitch is None:
            self._filtered_pitch = pitch
            self._filtered_yaw = yaw
            return pitch, yaw
        alpha = dt / (self._filter_tau_s + dt)
        self._filtered_pitch += alpha * (pitch - self._filtered_pitch)
        self._filtered_yaw += alpha * (yaw - self._filtered_yaw)
        return self._filtered_pitch, self._filtered_yaw

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _cable_delta_to_servo_norm(self, delta_L_mm: NDArray) -> NDArray:
        """缆长变化量 → 归一化舵机角度 [-1, 1].

        ΔL > 0 表示缆绳需放长，但舵机正角度对应缆绳缩短，
        因此取负号翻转方向。预紧偏置使中立位时缆绳略有拉力。
        """
        # 减去预紧量：中立位(ΔL=0)时舵机会轻微缩短，保持线缆绷紧
        biased = delta_L_mm - self._pretension_mm
        delta_theta_deg = biased / self._drum_radius * (180.0 / np.pi)
        norm = -delta_theta_deg / self.SERVO_HALF_DEG
        return norm
