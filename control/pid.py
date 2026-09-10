"""单轴 PID（行为级复刻 archive/example_code/steering_motor/Scripts/CloseLoop/pid_controller.py）。

- 一阶低通（alpha）→ 死区 + 静止判断 → 输出锁定（防抖）→ PID + 积分限幅。
- 积分/微分按"每拍"累计（不乘 dt），因此调用频率必须恒定（loop.py 为 100Hz）。
- 输出单位 = offset（int16，限幅 ±limit，默认 600，对应 example_code 的 LM）。
"""

from __future__ import annotations


class PIDController:
    def __init__(
        self,
        kp: float = 8.0,
        ki: float = 1.0,
        kd: float = 2.5,
        limit: float = 600.0,
        deadband: float = 0.2,
        alpha: float = 0.3,
        integral_range: float = 600.0,
        lock_velocity: float = 0.5,
        lock_count: int = 5,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.deadband = deadband
        self.alpha = alpha
        self.integral_range = integral_range
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
                raw = self.kp * error + self.ki * self.error_sum + \
                    self.kd * (error - self.last_error)
                self.locked_output = self._clamp(raw, self.limit)
                self.is_locked = True

        if self.is_locked:
            self.last_error = error
            return self.locked_output

        self.error_sum = self._clamp(self.error_sum + error, self.integral_range)
        d_error = error - self.last_error
        out = self.kp * error + self.ki * self.error_sum + self.kd * d_error
        self.last_error = error
        return self._clamp(out, self.limit)

    @staticmethod
    def _clamp(x: float, limit: float) -> float:
        return max(-limit, min(limit, x))
