"""直接监督训练前馈 g：Δq_actual → d_actual（差分空间逆映射）。

真值 = 开环数据里实际施加的差分命令 d_actual = [u1-u3, u2-u4]，
监督 = 实际实现的增量 Δq_actual = q[k+1] - q[k]（信号强，覆盖 0~5°）。

g 输入 = [Δq_actual(2), q_meas(2), gyro(3), force(4)]   (11 维)
g 输出 = 差分 d=(d1,d2)（2 维），损失 MSE(d_pred, d_actual)

部署：给定期望增量 Δq_d = v_d·dt，g 输出 d，decode(d, p) -> 4 路舵机角。
不依赖 F，直接监督，无 4->2 冗余问题。

用法：
    python -m feedforward.train_feedforward_direct \
      --data-dir <session> --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feedforward.dataset import (
    FORCE_COLS, IMU_COLS, align_sensors, load_sensor_df, load_servo_df,
)
from feedforward.feedforward_model import FeedforwardMLP

G_IN_DIM = 2 + 2 + 3 + 4   # Δq + q_meas + gyro + force


def build_direct(df):
    """展开 (Δq_actual, d_actual) 监督对。"""
    q = df[["current_pitch", "current_yaw"]].to_numpy(float)
    u = df[["servo_1_target_deg", "servo_2_target_deg",
            "servo_3_target_deg", "servo_4_target_deg"]].to_numpy(float)
    gyro = df[IMU_COLS].to_numpy(float)
    force = df[FORCE_COLS].to_numpy(float)

    ks = np.arange(0, len(q) - 1)
    dq = q[ks + 1] - q[ks]                    # (M,2) 实际增量
    d = u[ks, 0:2] - u[ks, 2:4]               # (M,2) 差分 [u1-u3, u2-u4]
    return {"dq": dq, "d": d, "q": q[ks], "gyro": gyro[ks], "force": force[ks]}


def to_t(x):
    return torch.tensor(np.asarray(x, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretension", type=float, default=0.15, help="部署时固定预紧（存进 checkpoint）")
    ap.add_argument("--val-mode", choices=["loso", "random"], default="random")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="feedforward/outputs/g_direct")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    df = load_servo_df(data_dir / "servo_data.csv")
    df = df[df["segment_id"] >= 0].reset_index(drop=True)
    imu = load_sensor_df(data_dir / "imu_data.csv", IMU_COLS)
    force = load_sensor_df(data_dir / "force_data.csv", FORCE_COLS)
    df = align_sensors(df, imu, force)

    ids = sorted(df["segment_id"].unique().tolist())
    if args.val_mode == "random":
        rng = np.random.default_rng(args.seed)
        order = list(rng.permutation(len(ids)))
        n_val = max(1, int(round(len(ids) * args.val_frac)))
        val_ids = set(ids[i] for i in order[:n_val])
    else:
        val_ids = {ids[-1]}
    tr_df = df[~df["segment_id"].isin(val_ids)].reset_index(drop=True)
    va_df = df[df["segment_id"].isin(val_ids)].reset_index(drop=True)

    tr = build_direct(tr_df)
    va = build_direct(va_df)

    # g 输入 + 归一化
    Xtr = np.concatenate([tr["dq"], tr["q"], tr["gyro"], tr["force"]], axis=1)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-9] = 1.0
    dtr = tr["d"]

    Xt = (to_t(Xtr) - to_t(mean)) / to_t(std)
    dt = to_t(dtr)
    dl = DataLoader(TensorDataset(Xt, dt), batch_size=512, shuffle=True)

    g = FeedforwardMLP(in_dim=G_IN_DIM).to(device)
    opt = torch.optim.Adam(g.parameters(), lr=args.lr)
    lossf = nn.MSELoss()

    for ep in range(args.epochs):
        g.train()
        total, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(g(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (ep + 1) % args.log_every == 0 or ep == args.epochs - 1:
            print(f"epoch {ep+1:4d}/{args.epochs}  loss={total/nb:.6f}")

    # 验证：差分预测误差
    Xva = np.concatenate([va["dq"], va["q"], va["gyro"], va["force"]], axis=1)
    Xv = (to_t(Xva) - to_t(mean)) / to_t(std)
    g.eval()
    with torch.no_grad():
        d_pred = g(Xv.to(device)).cpu().numpy()
    d_true = va["d"]
    err = d_pred - d_true
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    r2 = 1 - (rmse ** 2) / d_true.var(axis=0)
    print(f"val d MAE  (d1,d2) = {mae[0]:.4f}, {mae[1]:.4f} norm")
    print(f"val d RMSE (d1,d2) = {rmse[0]:.4f}, {rmse[1]:.4f} norm")
    print(f"val d R²   (d1,d2) = {r2[0]:.4f}, {r2[1]:.4f}")
    print(f"d 真实幅度 std  = {d_true.std(axis=0)[0]:.3f}, {d_true.std(axis=0)[1]:.3f} norm")

    # 线性前向：把 d 误差换算成"度"（粗略可解释性）
    def lin_fit(dx, dy):
        X = np.concatenate([dx, np.ones((len(dx), 1))], axis=1)
        return np.linalg.lstsq(X, dy, rcond=None)[0]

    c_p = lin_fit(tr["d"], tr["dq"][:, 0])
    c_y = lin_fit(tr["d"], tr["dq"][:, 1])

    def lin_pred(dx):
        X = np.concatenate([dx, np.ones((len(dx), 1))], axis=1)
        return np.stack([X @ c_p, X @ c_y], axis=1)

    dq_repro = lin_pred(d_pred)
    err_deg = dq_repro - va["dq"]
    print(f"线性前向复现 Δq MAE (pitch,yaw) = {np.abs(err_deg).mean(axis=0)[0]:.4f}, "
          f"{np.abs(err_deg).mean(axis=0)[1]:.4f} deg")

    torch.save({
        "model_state": g.state_dict(),
        "in_dim": G_IN_DIM,
        "g_mean": mean.tolist(),
        "g_std": std.tolist(),
        "pretension": args.pretension,
    }, out / "g_checkpoint.pt")
    metrics = {
        "val_d_mae_norm": mae.tolist(),
        "val_d_rmse_norm": rmse.tolist(),
        "val_d_r2": r2.tolist(),
        "val_repro_dq_mae_deg": np.abs(err_deg).mean(axis=0).tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"g checkpoint saved: {out / 'g_checkpoint.pt'}")
    print(f"outputs dir: {out}")


if __name__ == "__main__":
    main()
