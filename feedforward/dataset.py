"""把采集 CSV 组装成前向模型的窗口样本。

支持：
- q/u 的多步历史（默认 5 步）；
- IMU 陀螺（当前拍）与力传感器（当前拍）作为附加输入；
- 按 segment_id / trajectory_id 分段（leave-one-segment-out）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

U_COLS = ["servo_1_target_deg", "servo_2_target_deg",
          "servo_3_target_deg", "servo_4_target_deg"]
Q_COLS = ["current_pitch", "current_yaw"]
IMU_COLS = ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]
FORCE_COLS = ["ch1", "ch2", "ch3", "ch4"]


def load_servo_df(path):
    return pd.read_csv(path)


def load_sensor_df(path, cols):
    """加载传感器 CSV，仅保留时间戳 + 需要的列。"""
    df = pd.read_csv(path)
    keep = ["pc_timestamp_ns"] + [c for c in cols if c in df.columns]
    return df[keep]


def align_sensors(servo, imu=None, force=None):
    """把 IMU/力按时间戳对齐到 servo 行（causal：direction=backward 前向填充）。

    返回在 servo 上追加了 gyro_* / ch* 列的新 DataFrame。
    """
    df = servo.sort_values("pc_timestamp_ns").reset_index(drop=True)
    if imu is not None:
        s = imu.sort_values("pc_timestamp_ns")
        df = pd.merge_asof(df, s, on="pc_timestamp_ns", direction="backward")
        df[IMU_COLS] = df[IMU_COLS].ffill().bfill()
    if force is not None:
        s = force.sort_values("pc_timestamp_ns")
        df = pd.merge_asof(df, s, on="pc_timestamp_ns", direction="backward")
        df[FORCE_COLS] = df[FORCE_COLS].ffill().bfill()
    return df


def feature_dim(seq_len, use_imu=False, use_force=False):
    return seq_len * len(Q_COLS) + seq_len * len(U_COLS) \
        + (len(IMU_COLS) if use_imu else 0) \
        + (len(FORCE_COLS) if use_force else 0)


def segments_to_windows(df, seq_len=5, split_col="segment_id"):
    """按 split_col 分段，每段内部滑窗。

    X = [q 历史(seq_len 拍), u 历史(seq_len 拍), (可选)gyro 当前, (可选)force 当前]
    y = q_{k+1} - q_k

    IMU/力列在 df 中存在时自动纳入（gyro_*_dps / ch1..ch4）。
    """
    use_imu = all(c in df.columns for c in IMU_COLS)
    use_force = all(c in df.columns for c in FORCE_COLS)

    if split_col is not None and split_col in df.columns:
        groups = [(str(gid), g) for gid, g in df.groupby(split_col, sort=True)]
    else:
        groups = [("all", df)]

    segments = []
    for gid, g in groups:
        q = g[Q_COLS].to_numpy(dtype=float)
        u = g[U_COLS].to_numpy(dtype=float)
        gyro = g[IMU_COLS].to_numpy(dtype=float) if use_imu else None
        force = g[FORCE_COLS].to_numpy(dtype=float) if use_force else None
        if len(q) <= seq_len + 1:
            continue
        Xs, ys = [], []
        for k in range(seq_len - 1, len(q) - 1):
            feat = []
            for j in range(seq_len):
                feat.extend(q[k - j])
            for j in range(seq_len):
                feat.extend(u[k - j])
            if use_imu:
                feat.extend(gyro[k])
            if use_force:
                feat.extend(force[k])
            Xs.append(feat)
            ys.append(q[k + 1] - q[k])
        segments.append((gid, np.asarray(Xs, dtype=float), np.asarray(ys, dtype=float)))
    return segments


def normalize(train_segments, val_segments=None):
    """用训练段统计量做 z-score。返回 (scaler, train, val)。"""
    allX = np.concatenate([s[1] for s in train_segments], axis=0)
    mean = allX.mean(axis=0)
    std = allX.std(axis=0)
    std[std < 1e-9] = 1.0

    def apply(segs):
        return [(gid, (X - mean) / std, y) for gid, X, y in segs]

    train = apply(train_segments)
    val = apply(val_segments) if val_segments is not None else None
    return {"mean": mean, "std": std}, train, val
