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
import sys
from pathlib import Path

# 允许直接 `python feedforward/train.py` 运行（也支持 `python -m feedforward.train`）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feedforward import synthetic
from feedforward.dataset import (
    FORCE_COLS, IMU_COLS, align_sensors, feature_dim,
    load_sensor_df, load_servo_df, normalize, segments_to_windows,
)
from feedforward.model import ForwardMLP

SEQ_LEN = 5   # q/u 历史窗口步数


def to_torch(segments):
    return [(gid, torch.as_tensor(X, dtype=torch.float32),
             torch.as_tensor(y, dtype=torch.float32)) for gid, X, y in segments]


def train(model, train_ts, val_ts, epochs, lr, log_every=10, device="cpu"):
    model.to(device)
    X = torch.cat([x for _, x, _ in train_ts], dim=0)
    y = torch.cat([yy for _, _, yy in train_ts], dim=0)
    dl = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    history = {"epoch": [], "train_loss": [], "val_mae": [], "val_rmse": []}

    for ep in range(epochs):
        model.train()
        total, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        train_loss = total / nb

        mae, rmse = eval_onestep(model, val_ts, device)
        history["epoch"].append(ep + 1)
        history["train_loss"].append(train_loss)
        history["val_mae"].append(mae.tolist())
        history["val_rmse"].append(rmse.tolist())

        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"epoch {ep+1:4d}/{epochs}  loss={train_loss:.6f}  "
                  f"val MAE=({mae[0]:.4f},{mae[1]:.4f})  "
                  f"val RMSE=({rmse[0]:.4f},{rmse[1]:.4f})")

    return model, history


def eval_onestep(model, val_ts, device="cpu"):
    model.eval()
    errs = []
    with torch.no_grad():
        for _, X, y in val_ts:
            pred = model(X.to(device))
            errs.append((pred.cpu() - y).numpy())
    err = np.concatenate(errs, axis=0)
    return np.abs(err).mean(axis=0), np.sqrt((err ** 2).mean(axis=0))


def plot_history(history, out):
    """训练曲线：loss 与验证 MAE 随 epoch 变化。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = history["epoch"]
        mae = np.array(history["val_mae"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(ep, history["train_loss"], lw=1.2)
        axes[0].set_ylabel("train MSE loss")
        axes[0].set_yscale("log")
        axes[0].grid(alpha=0.3)
        axes[1].plot(ep, mae[:, 0], label="pitch MAE", lw=1.2)
        axes[1].plot(ep, mae[:, 1], label="yaw MAE", lw=1.2)
        axes[1].set_ylabel("val MAE (deg)")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.suptitle("Training curves")
        fig.tight_layout()
        png = out / "history.png"
        fig.savefig(png, dpi=110)
        print(f"history plot saved: {png}")
    except Exception as e:  # noqa: BLE001
        print(f"history plot skipped: {e}")


def rollout(model, scaler, df, seq_len=SEQ_LEN, split_col="trajectory_id", holdout=None, device="cpu"):
    """在留出轨迹上自回归 free-running。返回 (q_true, q_pred, t_s)。

    holdout: 留出的段 id（str 或 list[str]）；None 表示用整个 df。
    """
    model.eval()
    if split_col is not None and split_col in df.columns and holdout is not None:
        holdouts = [holdout] if isinstance(holdout, str) else list(holdout)
        g = df[df[split_col].astype(str).isin(holdouts)]
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
            x = torch.as_tensor((np.asarray(feat) - mean) / std, dtype=torch.float32).to(device)
            dq = model(x.unsqueeze(0)).squeeze(0).cpu().numpy()
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
    ap.add_argument("--log-every", type=int, default=10, help="每多少个 epoch 打印一次 loss/指标")
    ap.add_argument("--holdout", default=None, help="loso 模式留出的段 id")
    ap.add_argument("--val-mode", choices=["loso", "random"], default="loso",
                    help="验证切分：loso=留单段，random=段级随机抽取")
    ap.add_argument("--val-frac", type=float, default=0.25, help="random 模式验证段占比")
    ap.add_argument("--seed", type=int, default=0, help="random 模式随机种子")
    ap.add_argument("--no-sensors", action="store_true", help="不加载 IMU/力（消融：纯 q/u）")
    ap.add_argument("--out", default="feedforward/outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    use_imu = use_force = False
    if args.data_dir:
        data_dir = Path(args.data_dir)
        csv_path = data_dir / "servo_data.csv"
        imu = force = None
        if not args.no_sensors:
            if (data_dir / "imu_data.csv").exists():
                imu = load_sensor_df(data_dir / "imu_data.csv", IMU_COLS)
                use_imu = len(imu) > 0
            if (data_dir / "force_data.csv").exists():
                force = load_sensor_df(data_dir / "force_data.csv", FORCE_COLS)
                use_force = len(force) > 0
    else:
        csv_path = synthetic.generate_session(out / "synthetic", seed=0)
        imu = force = None

    df = load_servo_df(csv_path)
    # 段切分优先级：segment_id（开环扫描段，剔除 dwell/标定 segment_id<0）> trajectory_id > 时间切分
    if "segment_id" in df.columns:
        df = df[df["segment_id"] >= 0].reset_index(drop=True)
        split_col = "segment_id"
    elif "trajectory_id" in df.columns:
        split_col = "trajectory_id"
    else:
        split_col = None

    # 对齐 IMU/力到 servo（causal：direction=backward 前向填充）
    if imu is not None or force is not None:
        df = align_sensors(df, imu if use_imu else None, force if use_force else None)

    in_dim = feature_dim(SEQ_LEN, use_imu, use_force)
    print(f"features: seq_len={SEQ_LEN}, imu={use_imu}, force={use_force}, in_dim={in_dim}")
    segs = segments_to_windows(df, seq_len=SEQ_LEN, split_col=split_col)

    if split_col:
        ids = [s[0] for s in segs]
        if args.val_mode == "random":
            # 段级随机抽样：验证集是若干条整段（无泄漏），且分布接近平均
            rng = np.random.default_rng(args.seed)
            order = list(rng.permutation(len(ids)))
            n_val = max(1, int(round(len(ids) * args.val_frac)))
            val_ids = sorted(ids[i] for i in order[:n_val])
        else:  # loso：留单段（压力测试）
            val_ids = [args.holdout or ids[-1]]
        train_segs = [s for s in segs if s[0] not in val_ids]
        val_segs = [s for s in segs if s[0] in val_ids]
        holdout = ",".join(val_ids) if len(val_ids) <= 4 else f"{len(val_ids)}segs"
    else:
        # 真实单文件没有 segment/trajectory id：按时间 80/20 切分（rollout 覆盖全文件，仅供初步观察）
        seg = segs[0]
        n = len(seg[1])
        cut = int(n * 0.8)
        train_segs = [(seg[0], seg[1][:cut], seg[2][:cut])]
        val_segs = [(seg[0], seg[1][cut:], seg[2][cut:])]
        val_ids = None
        holdout = "tail_20%"

    scaler, train_segs_n, val_segs_n = normalize(train_segs, val_segs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = ForwardMLP(in_dim=in_dim)
    val_ts = to_torch(val_segs_n)
    model, history = train(model, to_torch(train_segs_n), val_ts,
                           args.epochs, args.lr, log_every=args.log_every, device=device)

    # 保存 F 权重 + 归一化器 + 特征配置，供 train_feedforward.py 加载
    torch.save({
        "model_state": model.state_dict(),
        "seq_len": SEQ_LEN,
        "use_imu": use_imu,
        "use_force": use_force,
        "in_dim": in_dim,
        "scaler_mean": scaler["mean"].tolist(),
        "scaler_std": scaler["std"].tolist(),
    }, out / "f_checkpoint.pt")
    print(f"F checkpoint saved: {out / 'f_checkpoint.pt'}")

    mae, rmse = eval_onestep(model, val_ts, device)
    print(f"holdout = {holdout}")
    print(f"1-step  Δq MAE  (pitch,yaw) = {mae[0]:.4f}, {mae[1]:.4f} deg")
    print(f"1-step  Δq RMSE (pitch,yaw) = {rmse[0]:.4f}, {rmse[1]:.4f} deg")

    # 训练曲线（loss / val MAE 随 epoch）
    np.savez(out / "history.npz",
             epoch=np.array(history["epoch"]),
             train_loss=np.array(history["train_loss"]),
             val_mae=np.array(history["val_mae"]),
             val_rmse=np.array(history["val_rmse"]))
    plot_history(history, out)

    metrics = {
        "holdout": str(holdout),
        "onestep_mae_deg": mae.tolist(),
        "onestep_rmse_deg": rmse.tolist(),
    }

    if use_imu or use_force:
        # F 含传感器输入，free-running 时拿不到真实 IMU/力，无法 rollout
        print("rollout skipped: F 含 IMU/力输入，free-running 无真实传感器")
    else:
        q_true, q_pred, t = rollout(model, scaler, df, split_col=split_col, holdout=val_ids, device=device)
        err = q_pred - q_true
        err = err[SEQ_LEN:]
        print(f"rollout MAE  (pitch,yaw) = {np.abs(err).mean(axis=0)[0]:.4f}, "
              f"{np.abs(err).mean(axis=0)[1]:.4f} deg")
        print(f"rollout RMSE (pitch,yaw) = {np.sqrt((err ** 2).mean(axis=0))[0]:.4f}, "
              f"{np.sqrt((err ** 2).mean(axis=0))[1]:.4f} deg")
        fde_axis = np.abs(err[-1])
        fde_euclid = float(np.hypot(fde_axis[0], fde_axis[1]))
        print(f"rollout FDE  (pitch,yaw) = {fde_axis[0]:.4f}, {fde_axis[1]:.4f} deg  "
              f"(euclidean {fde_euclid:.4f} deg)")
        np.savez(out / "rollout.npz", t=t, q_true=q_true, q_pred=q_pred)
        metrics.update({
            "rollout_mae_deg": np.abs(err).mean(axis=0).tolist(),
            "rollout_rmse_deg": np.sqrt((err ** 2).mean(axis=0)).tolist(),
            "rollout_fde_deg": fde_axis.tolist(),
            "rollout_fde_euclid_deg": fde_euclid,
        })
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

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"outputs dir: {out}")


if __name__ == "__main__":
    main()
