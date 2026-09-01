"""离线合成数据：模拟一个"带死区 + 轴间耦合"的 2-DOF 被控对象。

用它与真实格式一致的 servo_data.csv，跑通"数据 -> 前向模型 -> rollout 评估"整条链路。

模型侧本阶段不建模死区/静摩擦；在合成植物里放一个温和死区，正好用来观察朴素 F 的误差分布。

真实数据路径：真实采集时 servo_data.csv 已含 current_pitch/current_yaw（动捕解算姿态），
故前向模型第一版只读这一个文件即可，不需要动捕四元数解算。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from control_model.excitation import default_segments, sample_segment

FS = 100.0
DT = 1.0 / FS

SERVO_COLUMNS = [
    "pc_timestamp_ns", "pc_receive_unix_time_ms",
    "servo_1_target_deg", "servo_2_target_deg", "servo_3_target_deg", "servo_4_target_deg",
    "target_pitch", "target_yaw", "current_pitch", "current_yaw",
    "phase", "trajectory_id",
]


class SyntheticPlant:
    """极简 2-DOF 植物：差分 -> 死区 -> 线性耦合 -> 积分。

    G: 差分(归一化) -> 关节角速度(deg/s) 的 2x2 耦合增益。
    deadzone: 每轴差分死区（归一化），|d| < deadzone 时不产生运动。
    noise_deg: 关节角测量噪声标准差（模拟动捕噪声）。
    """

    def __init__(self, G=None, deadzone=0.05, noise_deg=0.05, q0=(0.0, 0.0), seed=0):
        self.G = G if G is not None else np.array([[120.0, 15.0], [-10.0, 110.0]])
        self.deadzone = deadzone
        self.noise_deg = noise_deg
        self.q = np.asarray(q0, dtype=float)
        self.rng = np.random.default_rng(seed)

    def step(self, u_norm):
        d = np.array([u_norm[0] - u_norm[2], u_norm[1] - u_norm[3]])
        d_eff = np.sign(d) * np.maximum(np.abs(d) - self.deadzone, 0.0)
        qdot = self.G @ d_eff
        self.q = self.q + qdot * DT
        return self.q + self.noise_deg * self.rng.standard_normal(2)


def generate_session(out_dir, seed=0):
    """生成一个合成会话的 servo_data.csv，返回文件路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plant = SyntheticPlant(seed=seed)
    rows = []
    t0_ns = 1_000_000_000

    for tid, seg in enumerate(default_segments()):
        t, u = sample_segment(seg, p=0.15, fs=FS)
        for i in range(len(t)):
            u_i = u[:, i]  # u 形状为 (4, N)，取第 i 列
            q = plant.step(u_i)
            rows.append({
                "pc_timestamp_ns": t0_ns + int(t[i] * 1e9),
                "pc_receive_unix_time_ms": int(t[i] * 1000),
                "servo_1_target_deg": u_i[0],
                "servo_2_target_deg": u_i[1],
                "servo_3_target_deg": u_i[2],
                "servo_4_target_deg": u_i[3],
                "target_pitch": 0.0,
                "target_yaw": 0.0,
                "current_pitch": q[0],
                "current_yaw": q[1],
                "phase": "exploration",
                "trajectory_id": str(tid),
            })

    df = pd.DataFrame(rows, columns=SERVO_COLUMNS)
    path = out_dir / "servo_data.csv"
    df.to_csv(path, index=False)
    return path
