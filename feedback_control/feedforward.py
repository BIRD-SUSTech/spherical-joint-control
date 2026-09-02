"""前馈补偿控制器（加载 g）+ 任务空间差分 PID。

g：Δq_d, q_meas, gyro, force -> 差分 d=(d1,d2)，解码(+预紧 p) -> 4 路归一化舵机角。
DiffPID：任务空间误差 -> 差分（归一化），作为反馈兜底。

单位约定（与 g 训练一致）：
    Δq_d / q_meas: 度；gyro: deg/s（IMU 原始轴）；force: 原始 ch1-4；d: 归一化差分。
"""

from __future__ import annotations

import numpy as np
import torch

from feedforward.feedforward_model import FeedforwardMLP


def decode(d, p):
    """差分 d=(d1,d2) + 预紧 p -> 4 路归一化舵机角 [-1,1]。"""
    d1, d2 = float(d[0]), float(d[1])
    n = np.array([p + d1 / 2.0, p + d2 / 2.0, p - d1 / 2.0, p - d2 / 2.0])
    return np.clip(n, -1.0, 1.0)


class FeedforwardController:
    """加载 g_checkpoint.pt，计算前馈指令 u_ff。"""

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self._g = FeedforwardMLP(in_dim=ckpt["in_dim"]).to(device)
        self._g.load_state_dict(ckpt["model_state"])
        self._g.eval()
        self._p = float(ckpt.get("pretension", 0.15))
        self._mean = np.asarray(ckpt["g_mean"], dtype=np.float32)
        self._std = np.asarray(ckpt["g_std"], dtype=np.float32)

    @property
    def pretension(self):
        return self._p

    def compute(self, dq_d, q_meas, gyro, force):
        """-> 差分 d (2,)。dq_d(2), q_meas(2), gyro(3), force(4)。"""
        x = np.concatenate([np.ravel(dq_d), np.ravel(q_meas),
                            np.ravel(gyro), np.ravel(force)]).astype(np.float32)
        x = (x - self._mean) / self._std
        xt = torch.from_numpy(x).to(self._device)
        with torch.no_grad():
            d = self._g(xt.unsqueeze(0)).squeeze(0).cpu().numpy()
        return d.astype(float)

    def u_ff(self, dq_d, q_meas, gyro, force):
        """-> 4 路归一化舵机角 [-1,1]。"""
        return decode(self.compute(dq_d, q_meas, gyro, force), self._p)


class DiffPID:
    """任务空间 PID -> 差分 d（归一化）。

    符号：实测 d1 正 -> pitch 负（pitch≈-gain*d1），故反馈差分取负号。
    kp 单位：归一化差分 / 度（参考增益 ~0.03~0.05）。
    """

    def __init__(self, kp_pitch=0.04, kp_yaw=0.04, kd_pitch=0.0, kd_yaw=0.0,
                 integral_max=0.2):
        self.kp_pitch, self.kp_yaw = kp_pitch, kp_yaw
        self.kd_pitch, self.kd_yaw = kd_pitch, kd_yaw
        self.integral_max = integral_max
        self.reset()

    def reset(self):
        self._i = np.zeros(2)
        self._prev = None

    def update(self, e_pitch, e_yaw, dt):
        e = np.array([e_pitch, e_yaw])
        self._i = np.clip(self._i + e * dt, -self.integral_max, self.integral_max)
        de = np.zeros(2) if self._prev is None else (e - self._prev) / max(dt, 1e-6)
        self._prev = e
        d = -(np.array([self.kp_pitch, self.kp_yaw]) * e
              + np.array([self.kd_pitch, self.kd_yaw]) * de + self._i)
        return d
