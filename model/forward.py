"""前向模型 F（M5 起步：线性 ARX F_v0）。

Δq = W @ [q 历史, u 历史, 方向特征, 换向特征, 1]
闭式最小二乘解，无需迭代训练。分段线性 F_v1 后续按残差指引再加。
"""

from __future__ import annotations

import numpy as np


def fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """最小二乘闭式解（含偏置项）。返回 W，形状 (in_dim+1, out_dim)。"""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    W, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return W


def predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """一步预测 Δq。"""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ W


def rollout(W: np.ndarray, q: np.ndarray, u: np.ndarray, seg: np.ndarray,
            seq_len: int = 5, start_frac: float = 0.0) -> np.ndarray:
    """free-running rollout：每段从 start_frac 起自回归，返回预测 q（与输入等长）。

    方向/换向特征在 rollout 时从 q_pred 逐步重算，since_reversal 按段内拍数维护。
    """
    n = len(q)
    q_pred = q.copy()

    for gid in np.unique(seg):
        idx = np.where(seg == gid)[0]
        n0 = len(idx)
        start = max(seq_len, int(start_frac * n0))
        last_rev = np.full(2, start, dtype=int)

        for i in range(start, n0 - 1):
            k = idx[i]
            # 当前拍 qdot（用 q_pred，前几拍可能已预测）
            dqk = q_pred[k] - q_pred[k - 1] if k >= 1 else np.zeros(2)
            # 维护段内 since_reversal（变号则更新 last_rev）
            if i > start:
                prev_dq = q_pred[k - 1] - q_pred[k - 2] if k >= 2 else np.zeros(2)
                for a in range(2):
                    if dqk[a] * prev_dq[a] < 0:
                        last_rev[a] = i
            since = np.minimum(i - last_rev, 20) / 20.0

            feat = []
            feat += q_pred[k - seq_len + 1:k + 1].reshape(-1).tolist()
            feat += u[k - seq_len + 1:k + 1].reshape(-1).tolist()
            feat += [float(np.sign(dqk[0])), float(np.sign(dqk[1]))]
            feat += [float(np.sign(u[k, 0])), float(np.sign(u[k, 1]))]
            feat += [since[0], since[1]]

            dq = predict(W, np.asarray(feat).reshape(1, -1))[0]
            q_pred[k + 1] = q_pred[k] + dq

    return q_pred


def fit_direction_segmented(X: np.ndarray, y: np.ndarray, u: np.ndarray) -> dict:
    """F_v1：按方向分段线性。

    每个输出轴（fb/lr）按对应输入轴 sign(u) 分正负两组，各拟合一个 W。
    返回 dict{"fb_pos": W, "fb_neg": W, "lr_pos": W, "lr_neg": W}。
    每个 W 形状 (in_dim+1, 2)，预测时取对应轴的输出列（保留交叉耦合）。
    """
    models = {}
    for out_axis, u_axis, name in [(0, 0, "fb"), (1, 1, "lr")]:
        for sign_val, label in [(1, "pos"), (-1, "neg")]:
            mask = np.sign(u[:, u_axis]) == sign_val
            if mask.sum() < 10:
                models[f"{name}_{label}"] = None
            else:
                models[f"{name}_{label}"] = fit_linear(X[mask], y[mask])
    return models


def predict_segmented(models: dict, X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """F_v1：按方向分段预测 Δq。"""
    y_hat = np.zeros((X.shape[0], 2))
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    for out_axis, u_axis, name in [(0, 0, "fb"), (1, 1, "lr")]:
        for sign_val, label in [(1, "pos"), (-1, "neg")]:
            mask = np.sign(u[:, u_axis]) == sign_val
            W = models[f"{name}_{label}"]
            if W is not None and mask.sum() > 0:
                y_hat[mask, out_axis] = (Xb[mask] @ W)[:, out_axis]
    return y_hat
