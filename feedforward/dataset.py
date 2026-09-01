"""把 servo_data.csv 组装成前向模型的窗口样本。"""

from __future__ import annotations

import numpy as np
import pandas as pd

U_COLS = ["servo_1_target_deg", "servo_2_target_deg",
          "servo_3_target_deg", "servo_4_target_deg"]
Q_COLS = ["current_pitch", "current_yaw"]


def load_servo_df(path):
    return pd.read_csv(path)


def segments_to_windows(df, seq_len=2, split_col="trajectory_id"):
    """按 split_col 分段，每段内部滑窗。

    样本：X = [q_k, q_{k-1}, ..., u_k, u_{k-1}, ...]（历史窗口）
          y = q_{k+1} - q_k（下一拍姿态增量）

    Returns:
        list[(segment_id, X (n, in_dim), y (n, 2))]
    """
    if split_col is not None and split_col in df.columns:
        groups = [(str(gid), g) for gid, g in df.groupby(split_col, sort=True)]
    else:
        groups = [("all", df)]

    segments = []
    for gid, g in groups:
        q = g[Q_COLS].to_numpy(dtype=float)
        u = g[U_COLS].to_numpy(dtype=float)
        if len(q) <= seq_len + 1:
            continue
        Xs, ys = [], []
        for k in range(seq_len - 1, len(q) - 1):
            feat = []
            for j in range(seq_len):
                feat.extend(q[k - j])
            for j in range(seq_len):
                feat.extend(u[k - j])
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
