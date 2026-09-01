"""最小训练/评估脚本（跑通全流程）。

用法：
    # 离线合成数据跑通全流程
    python -m feedforward.train --synthetic --epochs 300

    # 用真实采集的 servo_data.csv
    python -m feedforward.train --data-dir <session_dir> --epochs 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import synthetic
from .dataset import load_servo_df, segments_to_windows, normalize
from .model import ForwardMLP

SEQ_LEN = 2
IN_DIM = SEQ_LEN * 2 + SEQ_LEN * 4   # q 历史(2*seq) + u 历史(4*seq)


def to_torch(segments):
    return [(gid, torch.as_tensor(X, dtype=torch.float32),
             torch.as_tensor(y, dtype=torch.float32)) for gid, X, y in segments]


def train(model, train_ts, epochs, lr):
    X = torch.cat([x for _, x, _ in train_ts], dim=0)
    y = torch.cat([yy for _, _, yy in train_ts], dim=0)
    dl = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def eval_onestep(model, val_ts):
    model.eval()
    errs = []
    with torch.no_grad():
        for _, X, y in val_ts:
            pred = model(X)
            errs.append((pred - y).numpy())
    err = np.concatenate(errs, axis=0)
    return np.abs(err).mean(axis=0), np.sqrt((err ** 2).mean(axis=0))


def rollout(model, scaler, df, seq_len=SEQ_LEN, split_col="trajectory_id", holdout=None):
    """在留出轨迹上自回归 free-running。返回 (q_true, q_pred, t_s)。"""
    model.eval()
    if split_col is not None and split_col in df.columns:
        g = df[df[split_col].astype(str) == str(holdout)]
    else:
        g = df
    q = g[["current_pitch", "current_yaw"]].to_numpy(dtype=float)
    u = g[["servo_1_target_deg", "servo_2_target_deg",
           "servo_3_target_deg", "servo_4_target_deg"]].to_numpy(dtype=float)
    mean = scaler["mean"]
    std = scaler["std"]

    q_hist = q.copy()
    with torch.no_grad():
        for k in range(seq_len - 1, len(q) - 1):
            feat = []
            for j in range(seq_len):
                feat.extend(q_hist[k - j])
            for j in range(seq_len):
                feat.extend(u[k - j])
            x = torch.as_tensor((np.asarray(feat) - mean) / std, dtype=torch.float32)
            dq = model(x.unsqueeze(0)).squeeze(0).numpy()
            q_hist[k + 1] = q_hist[k] + dq

    t = g["pc_timestamp_ns"].to_numpy(dtype=float)
    t = (t - t[0]) / 1e9
    return q, q_hist, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="用合成数据跑通流程")
    ap.add_argument("--data-dir", default=None, help="含 servo_data.csv 的会话目录")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", default=None, help="留出的 trajectory_id")
    ap.add_argument("--out", default="feedforward/outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        csv_path = Path(args.data_dir) / "servo_data.csv"
    else:
        csv_path = synthetic.generate_session(out / "synthetic", seed=0)

    df = load_servo_df(csv_path)
    split_col = "trajectory_id" if "trajectory_id" in df.columns else None
    segs = segments_to_windows(df, seq_len=SEQ_LEN, split_col=split_col)

    if split_col:
        ids = [s[0] for s in segs]
        holdout = args.holdout or ids[-1]
        train_segs = [s for s in segs if s[0] != holdout]
        val_segs = [s for s in segs if s[0] == holdout]
    else:
        # 真实单文件没有 trajectory_id：按时间 80/20 切分（rollout 覆盖全文件，仅供初步观察）
        seg = segs[0]
        n = len(seg[1])
        cut = int(n * 0.8)
        train_segs = [(seg[0], seg[1][:cut], seg[2][:cut])]
        val_segs = [(seg[0], seg[1][cut:], seg[2][cut:])]
        holdout = "tail_20%"

    scaler, train_segs_n, val_segs_n = normalize(train_segs, val_segs)

    model = ForwardMLP(in_dim=IN_DIM)
    train(model, to_torch(train_segs_n), args.epochs, args.lr)

    mae, rmse = eval_onestep(model, to_torch(val_segs_n))
    print(f"holdout = {holdout}")
    print(f"1-step  Δq MAE  (pitch,yaw) = {mae[0]:.4f}, {mae[1]:.4f} deg")
    print(f"1-step  Δq RMSE (pitch,yaw) = {rmse[0]:.4f}, {rmse[1]:.4f} deg")

    q_true, q_pred, t = rollout(model, scaler, df, split_col=split_col, holdout=holdout)
    err = q_pred - q_true
    err = err[SEQ_LEN:]
    print(f"rollout MAE  (pitch,yaw) = {np.abs(err).mean(axis=0)[0]:.4f}, "
          f"{np.abs(err).mean(axis=0)[1]:.4f} deg")
    print(f"rollout RMSE (pitch,yaw) = {np.sqrt((err ** 2).mean(axis=0))[0]:.4f}, "
          f"{np.sqrt((err ** 2).mean(axis=0))[1]:.4f} deg")

    np.savez(out / "rollout.npz", t=t, q_true=q_true, q_pred=q_pred)
    metrics = {
        "holdout": str(holdout),
        "onestep_mae_deg": mae.tolist(),
        "onestep_rmse_deg": rmse.tolist(),
        "rollout_mae_deg": np.abs(err).mean(axis=0).tolist(),
        "rollout_rmse_deg": np.sqrt((err ** 2).mean(axis=0)).tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for ax, axis in zip(axes, ("pitch", "yaw")):
            i = 0 if axis == "pitch" else 1
            ax.plot(t, q_true[:, i], label="true", lw=1.4)
            ax.plot(t, q_pred[:, i], label="rollout", lw=1.0, alpha=0.8)
            ax.set_ylabel(f"{axis} (deg)")
            ax.legend()
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("t (s)")
        fig.suptitle(f"Forward model rollout (holdout={holdout})")
        fig.tight_layout()
        png = out / "rollout.png"
        fig.savefig(png, dpi=110)
        print(f"plot saved: {png}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")

    print(f"outputs dir: {out}")


if __name__ == "__main__":
    main()
