"""L4 数据 pipeline：读统一 schema 会话 → 窗口样本 + 物理特征（M5）。

输入：servo_data.csv（M2 统一 schema）
    u = servo_front_back_offset / servo_left_right_offset（2 路差分 offset）
    q = current_front_back_deg / current_left_right_deg（动捕实测，度）
    segment_id → 分段（leave-one-segment-out）

窗口样本：
    X = [q 历史(seq_len 拍), u 历史(seq_len 拍), 方向特征, 换向特征]
    y = Δq = q_{k+1} − q_k

物理特征（设计文档 §7.3，显式注入方向/换向先验，权重由数据决定）：
    - sign(q̇)：当前拍 q 差分方向（±1）
    - sign(u)：当前拍 u 方向（±1）
    - since_reversal：距上次 q̇ 变号的拍数（归一化到 [0,1]，除以窗口长度）
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

Q_COLS = ["current_front_back_deg", "current_left_right_deg"]
U_COLS = ["servo_front_back_offset", "servo_left_right_offset"]


def load_session(servo_csv: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读 servo_data.csv → (q, u, seg)。q/u 形状 (n,2)，seg (n,)。"""
    rows = list(csv.DictReader(open(servo_csv, newline="", encoding="utf-8")))
    q = np.array([[float(r[Q_COLS[0]]), float(r[Q_COLS[1]])] for r in rows])
    u = np.array([[float(r[U_COLS[0]]), float(r[U_COLS[1]])] for r in rows])
    seg = np.array([int(r["segment_id"]) for r in rows])
    return q, u, seg


def compute_qdot(q: np.ndarray) -> np.ndarray:
    """q̇（一拍差分，首拍补 0）。"""
    qdot = np.vstack([np.zeros((1, 2)), np.diff(q, axis=0)])
    return qdot


def compute_since_reversal(qdot: np.ndarray, normalize_by: int = 20) -> np.ndarray:
    """距上次变号的拍数（每轴独立），除以 normalize_by 归一化到 [0,1]。"""
    n = qdot.shape[0]
    since = np.zeros((n, 2))
    last_rev = np.zeros(2, dtype=int)
    for k in range(1, n):
        for a in range(2):
            if qdot[k, a] * qdot[k - 1, a] < 0:
                last_rev[a] = k
        since[k] = np.minimum(k - last_rev, normalize_by) / normalize_by
    return since


def build_windows(q: np.ndarray, u: np.ndarray, seg: np.ndarray,
                  seq_len: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 segment 滑窗 → (X, y, seg_ids)。

    X 维度 = seq_len*2(q) + seq_len*2(u) + 2(sign q̇) + 2(sign u) + 2(since_reversal)。
    """
    qdot = compute_qdot(q)
    since_rev = compute_since_reversal(qdot)

    Xs, ys, sids = [], [], []
    qks, uks, qdks, srks = [], [], [], []
    for gid in np.unique(seg):
        idx = np.where(seg == gid)[0]
        qg, ug = q[idx], u[idx]
        qdg, srg = qdot[idx], since_rev[idx]
        n = len(idx)
        for k in range(seq_len - 1, n - 1):
            feat = []
            feat += qg[k - seq_len + 1:k + 1].reshape(-1).tolist()
            feat += ug[k - seq_len + 1:k + 1].reshape(-1).tolist()
            feat += [float(np.sign(qdg[k, 0])), float(np.sign(qdg[k, 1]))]
            feat += [float(np.sign(ug[k, 0])), float(np.sign(ug[k, 1]))]
            feat += [srg[k, 0], srg[k, 1]]
            Xs.append(feat)
            ys.append(qg[k + 1] - qg[k])
            sids.append(gid)
            qks.append(qg[k])
            uks.append(ug[k])
            qdks.append(qdg[k])
            srks.append(srg[k])
    return (np.asarray(Xs), np.asarray(ys), np.asarray(sids),
            np.asarray(qks), np.asarray(uks), np.asarray(qdks), np.asarray(srks))


def feature_dim(seq_len: int = 5) -> int:
    return seq_len * 2 + seq_len * 2 + 2 + 2 + 2


def feature_names(seq_len: int = 5) -> list[str]:
    names = []
    for j in range(seq_len - 1, -1, -1):
        names += [f"q_fb_t-{j}", f"q_lr_t-{j}"]
    for j in range(seq_len - 1, -1, -1):
        names += [f"u_fb_t-{j}", f"u_lr_t-{j}"]
    names += ["sign_qdot_fb", "sign_qdot_lr", "sign_u_fb", "sign_u_lr",
              "since_rev_fb", "since_rev_lr"]
    return names
