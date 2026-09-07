"""M5 训练/评估入口：线性 ARX（F_v0）+ 残差结构分解。

用法：
    python -m model.train --session <session_dir> [--seq-len 5] [--holdout 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from model.dataset import build_windows, feature_names, load_session
from model.forward import fit_linear, predict, rollout
from model.residual import decompose, format_report


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 前向模型训练 + 残差分解")
    ap.add_argument("--session", required=True, help="会话目录（含 servo_data.csv）")
    ap.add_argument("--seq-len", type=int, default=5, help="q/u 历史窗口步数")
    ap.add_argument("--holdout", type=int, default=None, help="留出段 id（缺省=最后一段）")
    ap.add_argument("--rollout", action="store_true", help="额外跑 free-running rollout")
    args = ap.parse_args()

    session = Path(args.session)
    csv_path = session / "servo_data.csv" if session.is_dir() else session
    if not csv_path.exists():
        print(f"文件不存在: {csv_path}", file=sys.stderr)
        return 1

    q, u, seg = load_session(csv_path)
    X, y, seg_ids, q_k, u_k, qdot_k, since_k = build_windows(q, u, seg, args.seq_len)
    print(f"样本: {X.shape[0]} 窗口, 特征维度 {X.shape[1]}")
    print(f"段: {sorted(np.unique(seg_ids).tolist())}")

    # leave-one-segment-out
    holdout = args.holdout if args.holdout is not None else int(seg_ids.max())
    tr = seg_ids != holdout
    va = seg_ids == holdout

    W = fit_linear(X[tr], y[tr])

    # 1-step 残差（留出段）
    y_hat = predict(W, X[va])
    e = y[va] - y_hat
    mae = np.abs(e).mean(axis=0)
    rmse = np.sqrt((e ** 2).mean(axis=0))
    print(f"\nholdout=seg{holdout}, 1-step MAE fb={mae[0]:.4f}° lr={mae[1]:.4f}°  "
          f"RMSE fb={rmse[0]:.4f}° lr={rmse[1]:.4f}°")

    # 残差分解（留出段）
    results = decompose(e, q_k[va], u_k[va], qdot_k[va], since_k[va])
    print()
    print(format_report(results, "fb"))
    print()
    print(format_report(results, "lr"))

    # 可选 rollout
    if args.rollout:
        q_pred = rollout(W, q, u, seg, args.seq_len)
        err = q_pred - q
        # 只统计留出段的 rollout 误差（后 seq_len 起）
        print(f"\nrollout MAE fb={np.abs(err[:, 0]).mean():.4f}° "
              f"lr={np.abs(err[:, 1]).mean():.4f}°")

    return 0


if __name__ == "__main__":
    sys.exit(main())
