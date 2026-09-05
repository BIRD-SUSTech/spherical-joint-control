"""单轴 PID 控制器（移植自 CloseLoop/pid_controller.py，保留其行为）。

与参考代码一致：一阶低通 → 死区/静止判断 → 输出锁定（防抖）→ PID + 积分限幅。
差异：输出单位从参考的 PWM 偏移(±400) 改为舵机角度(°, 默认 ±135)，增益需实机整定。

注意：积分/微分按“每拍”累计（不乘 dt），因此调用频率必须恒定（main 中为 100Hz）。
"""

from __future__ import annotations


class PIDController:
    def __init__(
        self,
        kp: float = 3.0,          # 起始值（参考 KP=8 是在 ±400 单位下，换算到 ±135 约 2.7）
        ki: float = 0.3,
        kd: float = 0.8,
        limit: float = 135.0,     # 输出限幅（舵机角度°, 半行程 ±135）
        deadband: float = 0.2,    # 死区（姿态角°）
        alpha: float = 0.3,       # 输入低通系数（新数据占比）
        integral_max: float = 135.0,
        lock_velocity: float = 0.5,
        lock_count: int = 5,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.deadband = deadband
        self.alpha = alpha
        self.integral_max = integral_max
        self.lock_velocity = lock_velocity
        self.lock_count = lock_count

        self.target = 0.0
        self.error_sum = 0.0
        self.last_error = 0.0
        self.filtered = 0.0
        self.locked_output = 0.0
        self.is_locked = False
        self._lock_counter = 0

    def reset(self) -> None:
        self.error_sum = 0.0
        self.last_error = 0.0
        self.filtered = 0.0
        self.locked_output = 0.0
        self.is_locked = False
        self._lock_counter = 0

    def calculate(self, current: float) -> float:
        # 一阶低通
        self.filtered = (1.0 - self.alpha) * self.filtered + self.alpha * current
        error = self.target - self.filtered

        in_deadband = abs(error) < self.deadband
        is_static = abs(error - self.last_error) < self.lock_velocity

        if in_deadband and is_static:
            self._lock_counter += 1
        else:
            self._lock_counter = 0
            self.is_locked = False

        if self._lock_counter > self.lock_count:
            if not self.is_locked:
                raw = self._raw_output(error)
                self.locked_output = self._clamp(raw, self.limit)
                self.is_locked = True

        if self.is_locked:
            self.last_error = error
            return self.locked_output

        self.error_sum = self._clamp(self.error_sum + error, self.integral_max)
        out = self._raw_output(error)
        self.last_error = error
        return self._clamp(out, self.limit)

    def _raw_output(self, error: float) -> float:
        d_error = error - self.last_error
        return self.kp * error + self.ki * self.error_sum + self.kd * d_error

    @staticmethod
    def _clamp(x: float, limit: float) -> float:
        return max(-limit, min(limit, x))
