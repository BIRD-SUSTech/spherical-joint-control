"""训练前馈补偿模型 g：经冻结的 1-step 仿真器 F 反传。

目标构造（速度空间稠密采样，非单一参考轨迹）：
    Δq_d = 方向(全方向均匀) × 速度幅值([0,v_max]均匀) × dt
    g 输入 = [Δq_d(2), q_meas(2), gyro(3), force(4)]   (11 维)
    g 输出 = 差分 d=(d1,d2) -> 解码(+固定预紧 p) -> 4 路舵机角

训练（teacher-forced 1-step，F 冻结，每 epoch 重采样 Δq_d 做数据增强）：
    u_ff      = decode(g(...), p)
    dq_pred   = F(q_hist, u_ff, u_hist_past, gyro, force)   # F 预测这一拍增量
    L         = MSE(dq_pred, Δq_d)

真值：没有"正确 u"；用期望增量 Δq_d 当损失目标，F 提供 ∂q/∂u 反传。

用法：
    python -m feedforward.train_feedforward \
      --data-dir <session> --f-checkpoint <f_checkpoint.pt> --epochs 200
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
from feedforward.model import ForwardMLP

SEQ_LEN = 5
G_IN_DIM = 2 + 2 + 3 + 4   # Δq_d + q_meas + gyro + force


def decode_torch(d, p):
    """差分 d=(...,2) + 预紧 p -> 4 路归一化舵机角 [-1,1]。"""
    d1, d2 = d[..., 0:1], d[..., 1:2]
    return torch.cat([p + d1 / 2.0, p + d2 / 2.0, p - d1 / 2.0, p - d2 / 2.0], dim=-1)


def sample_desired_increments(n, rng, v_max, dt, v_min=0.0):
    """稠密采样期望增量 Δq_d：方向全向均匀 × 速度幅值 [v_min,v_max] 均匀。"""
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    v = rng.uniform(v_min, v_max, n)
    return np.stack([v * np.cos(theta), v * np.sin(theta)], axis=1) * dt


def build_arrays(df, seq_len=SEQ_LEN):
    """把对齐后的 df 展开成滑窗数组（固定状态部分）。"""
    q = df[["current_pitch", "current_yaw"]].to_numpy(float)
    u = df[["servo_1_target_deg", "servo_2_target_deg",
            "servo_3_target_deg", "servo_4_target_deg"]].to_numpy(float)
    gyro = df[IMU_COLS].to_numpy(float)
    force = df[FORCE_COLS].to_numpy(float)

    ks = np.arange(seq_len - 1, len(q) - 1)
    q_hist = np.concatenate([q[ks - j] for j in range(seq_len)], axis=1)          # 最近在前
    u_hist_past = np.concatenate([u[ks - j] for j in range(1, seq_len)], axis=1)  # u[k-1..k-4]
    return {
        "q_hist": q_hist,
        "u_hist_past": u_hist_past,
        "gyro": gyro[ks],
        "force": force[ks],
        "q_meas": q[ks],
    }


def to_t(x):
    return torch.tensor(np.asarray(x, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--f-checkpoint", required=True, help="F 权重 f_checkpoint.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretension", type=float, default=0.15, help="固定共模预紧 p")
    ap.add_argument("--v-max", type=float, default=25.0, help="期望速度幅值上界 (°/s)")
    ap.add_argument("--dt", type=float, default=0.01, help="控制步长 (s)")
    ap.add_argument("--val-mode", choices=["loso", "random"], default="random")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="feedforward/outputs/g")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 加载 F（冻结）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    ckpt = torch.load(args.f_checkpoint, map_location="cpu", weights_only=False)
    F = ForwardMLP(in_dim=ckpt["in_dim"]).to(device)
    F.load_state_dict(ckpt["model_state"])
    F.eval()
    for p in F.parameters():
        p.requires_grad_(False)
    f_mean = to_t(ckpt["scaler_mean"]).to(device)
    f_std = to_t(ckpt["scaler_std"]).to(device)
    print(f"F loaded: in_dim={ckpt['in_dim']}, imu={ckpt['use_imu']}, force={ckpt['use_force']}")

    # 2. 数据 + 对齐 + 分段切分
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

    tr = build_arrays(tr_df)
    va = build_arrays(va_df)

    # 3. 固定状态张量（训练）
    q_meas_t = to_t(tr["q_meas"]).to(device)
    gyro_t = to_t(tr["gyro"]).to(device)
    force_t = to_t(tr["force"]).to(device)
    q_hist_t = to_t(tr["q_hist"]).to(device)
    u_hist_past_t = to_t(tr["u_hist_past"]).to(device)
    M = len(tr["q_meas"])

    # g 输入归一化器（用固定状态 + 代表性 Δq_d 样本）
    _rng = np.random.default_rng(args.seed)
    _dq0 = sample_desired_increments(M, _rng, args.v_max, args.dt)
    g_in0 = np.concatenate([_dq0, tr["q_meas"], tr["gyro"], tr["force"]], axis=1)
    g_mean = to_t(g_in0.mean(axis=0)).to(device)
    g_std = to_t(g_in0.std(axis=0)).to(device)
    g_std[g_std < 1e-9] = 1.0

    # 4. 训练 g（每 epoch 重采样 Δq_d）
    g = FeedforwardMLP(in_dim=G_IN_DIM).to(device)
    opt = torch.optim.Adam(g.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    p = args.pretension
    rng = np.random.default_rng(args.seed + 1)

    for ep in range(args.epochs):
        g.train()
        dq_d = to_t(sample_desired_increments(M, rng, args.v_max, args.dt)).to(device)
        g_in = torch.cat([dq_d, q_meas_t, gyro_t, force_t], dim=1)
        g_in = (g_in - g_mean) / g_std
        ds = TensorDataset(g_in, dq_d, q_hist_t, u_hist_past_t, gyro_t, force_t)
        dl = DataLoader(ds, batch_size=512, shuffle=True)

        total, nb = 0.0, 0
        for xb, dqb, qhb, uhb, gyb, fob in dl:
            opt.zero_grad()
            d = g(xb)
            u_ff = decode_torch(d, p)
            f_feat = torch.cat([qhb, u_ff, uhb, gyb, fob], dim=1)
            f_feat = (f_feat - f_mean) / f_std
            dq_pred = F(f_feat)
            loss = lossf(dq_pred, dqb)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (ep + 1) % args.log_every == 0 or ep == args.epochs - 1:
            print(f"epoch {ep+1:4d}/{args.epochs}  loss={total/nb:.6f}")

    # 5. 验证：期望增量跟踪误差 |dq_pred - Δq_d|
    g.eval()
    rng2 = np.random.default_rng(args.seed + 100)
    dq_d_va = to_t(sample_desired_increments(len(va["q_meas"]), rng2, args.v_max, args.dt)).to(device)
    g_in_va = torch.cat([dq_d_va, to_t(va["q_meas"]).to(device), to_t(va["gyro"]).to(device),
                         to_t(va["force"]).to(device)], dim=1)
    g_in_va = (g_in_va - g_mean) / g_std
    with torch.no_grad():
        d = g(g_in_va)
        u_ff = decode_torch(d, p)
        f_feat = torch.cat([to_t(va["q_hist"]).to(device), u_ff, to_t(va["u_hist_past"]).to(device),
                            to_t(va["gyro"]).to(device), to_t(va["force"]).to(device)], dim=1)
        f_feat = (f_feat - f_mean) / f_std
        dq_pred = F(f_feat)
        err = (dq_pred - dq_d_va).cpu().numpy()

    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    mean_target = np.abs(dq_d_va.cpu().numpy()).mean(axis=0)   # "不动作"基线 ≈ mean|Δq_d|
    print(f"val Δq 跟踪 MAE  (pitch,yaw) = {mae[0]:.4f}, {mae[1]:.4f} deg")
    print(f"val Δq 跟踪 RMSE (pitch,yaw) = {rmse[0]:.4f}, {rmse[1]:.4f} deg")
    print(f"目标幅度 mean|Δq_d| (pitch,yaw) = {mean_target[0]:.4f}, {mean_target[1]:.4f} deg")

    # 6. 保存 g
    torch.save({
        "model_state": g.state_dict(),
        "in_dim": G_IN_DIM,
        "g_mean": g_mean.cpu().numpy().tolist(),
        "g_std": g_std.cpu().numpy().tolist(),
        "pretension": p,
    }, out / "g_checkpoint.pt")
    metrics = {
        "val_track_mae_deg": mae.tolist(),
        "val_track_rmse_deg": rmse.tolist(),
        "mean_target_dq_deg": mean_target.tolist(),
        "pretension": p,
        "v_max": args.v_max,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"g checkpoint saved: {out / 'g_checkpoint.pt'}")
    print(f"outputs dir: {out}")


if __name__ == "__main__":
    main()
