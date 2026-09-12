"""§8.4 反解 F 模式：前向时序模型 F（GRU）+ 运行时反解。

**架构**（纯模型，不用解析动态项）：
    h_t   = GRU( x_{t-L+1..t} )                     # 历史编码（含状态与传感器历史）
    Δq̇_t  = head( h_t , u_t )                       # 头显式吃【当前动作】u_t
    F 预测 Δq̇_t = q̇_{t+1} − q̇_t（∝ 加速度）

为什么预测 Δq̇ 而不是 Δq：
    一步 Δq ≈ q̇·dt 由**动量主导**，对 u 的灵敏度极小（∂Δq/∂u ~ 1e-4）→ 反解病态、噪声放大千倍。
    而 ∂Δq̇/∂u ≈ ω_n²·γ ≈ 10 °/s²/offset **良态**；且反解目标 `q̈_d` 来自**解析轨迹**（零噪声）。
    物理依据：`m q̈ = k(γu − q) − c q̇` ⟹ Δq̇ 对 u **线性**（头因此做成 affine-in-u，反解解析）。

**输入特征（每拍 16 维）**：q(2) q̇(2) u_prev(2) | IMU 陀螺(3) 加计(3) | 张力 ch1–ch4(4)
    —— IMU/张力的轴映射与特征工程**交给模型自学**（不手工设计，符合"纯模型"要求）。

**数据**：开环（u 为自变量，无闭环偏置，§9.2）+ 闭环（状态覆盖），按会话 LOSO 留出。

用法：
    python -m model.fit_forward_seq --session <A> <B> ... \
        --out configs/forward_seq_v1.json --epochs 60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

FEATURES = ["q_fb", "q_lr", "qdot_fb", "qdot_lr", "u_fb_prev", "u_lr_prev",
            "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
            "ch1", "ch2", "ch3", "ch4"]
N_FEAT = len(FEATURES)


# ---------------------------------------------------------------------------
# 数据：四路按时间戳对齐 → 逐拍特征 + Δq̇
# ---------------------------------------------------------------------------

def _rows(path: Path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _align(rows, cols):
    ts = np.array([int(r["pc_receive_unix_time_ms"]) for r in rows])
    vals = np.array([[float(r[c]) for c in cols] for r in rows])
    o = np.argsort(ts)
    return ts[o], vals[o]


def _smooth(x, w):
    """滑动均值（w 拍）——压动捕微分噪声。"""
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def _d1(x, dt, h):
    """中心一阶差分（±h 拍）。"""
    v = np.zeros_like(x)
    v[h:-h] = (x[2 * h:] - x[:-2 * h]) / (2 * h * dt)
    v[:h] = v[h]; v[-h:] = v[-h - 1]
    return v


def _d2(x, dt, h):
    """中心二阶差分（±h 拍）→ 加速度估计（低噪声版）。"""
    a = np.zeros_like(x)
    a[h:-h] = (x[2 * h:] - 2 * x[h:-h] + x[:-2 * h]) / (h * dt) ** 2
    a[:h] = a[h]; a[-h:] = a[-h - 1]
    return a


def _poly_pinv(w, dt):
    """因果零滞后最小二乘核：(3,w)，行 = [q, q̇, q̈] 在 τ=0 处的估计。

    对【过去 w 拍】拟合 q(τ)=a+bτ+cτ²/2（τ≤0 为过去），取 τ=0 处系数。
    因果、零相位滞后；代价是噪声增益随 w 增大。
    """
    j = np.arange(w)
    tau = (j - (w - 1)) * dt
    A = np.stack([np.ones(w), tau, tau ** 2 / 2], axis=1)
    return np.linalg.pinv(A)


def _poly_est(x, dt, w):
    """→ (q_est, qdot_est, qddot_est)；前 w-1 拍用首个有效值填充（启动瞬态）。"""
    M = _poly_pinv(w, dt)
    win = np.lib.stride_tricks.sliding_window_view(x, w)
    e = win @ M.T
    out = np.empty((len(x), 3))
    out[w - 1:] = e
    out[:w - 1] = e[0]
    return out[:, 0], out[:, 1], out[:, 2]


def load_session(session: Path, smooth: int = 15, acc_h: int = 10,
                 vel_mode: str = "central", vel_w: int = 31, with_aux: bool = False,
                 use_u_hist: bool = True, horizon: int = 0):
    """→ 逐拍 (X (n,16), Y (n,2)=Δq̇, seg_id (n,))；四路最近邻对齐到 servo 时钟。"""
    srows = [r for r in (_rows(session / "servo_data.csv") or [])
             if r.get("segment_id", "") not in ("", "-1", None)]
    if len(srows) < 200:
        return None
    per_seg: dict[int, list] = {}
    for r in srows:
        per_seg.setdefault(int(r["segment_id"]), []).append(r)

    imu = _align(*_rows_cols(session / "imu_data.csv",
                             ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
                              "ax_no_g_mps2", "ay_no_g_mps2", "az_no_g_mps2"]))
    fo = _align(*_rows_cols(session / "force_data.csv", ["ch1", "ch2", "ch3", "ch4"]))

    X, Y, S = [], [], []
    AUX = {"q": [], "q_d": [], "u": [], "qddot_causal": [], "sid": [], "t": []}
    for sid, rows in per_seg.items():
        rows.sort(key=lambda r: float(r["t_s"]))
        if len(rows) < 60:
            continue
        ts = np.array([int(r["pc_receive_unix_time_ms"]) for r in rows])
        st = np.array([float(r["t_s"]) for r in rows])
        qf = np.array([float(r["current_front_back_deg"]) for r in rows])
        ql = np.array([float(r["current_left_right_deg"]) for r in rows])
        uf = np.array([float(r["servo_front_back_offset"]) for r in rows])
        ul = np.array([float(r["servo_left_right_offset"]) for r in rows])
        n = len(rows)
        dt = np.median(np.diff(st))
        # 【关键】先平滑再差分：动捕 0.06°/拍噪声经两次数分会放大到 ~15 °/s²，
        # 与真实加速度（~30 °/s²）同量级 → 直接 np.gradient 两次得到的目标是噪声主导、学不动。
        qf_s, ql_s = _smooth(qf, smooth), _smooth(ql, smooth)
        # 标签 q̈：中心差分（低噪声、零相位差）。只用于离线教学，运行时不需要估计。
        af = _d2(qf_s, dt, acc_h); al = _d2(ql_s, dt, acc_h)
        # 特征 q̇：必须因果（运行时无法取未来）。central=中心差分（仅对照用，非因果）；
        # poly=因果零滞后多项式核。
        if vel_mode == "poly":
            pe_f, pe_l = _poly_est(qf, dt, vel_w), _poly_est(ql, dt, vel_w)
            vf, vl = pe_f[1], pe_l[1]
            acf, acl = pe_f[2], pe_l[2]
        else:
            vf = _d1(qf_s, dt, acc_h); vl = _d1(ql_s, dt, acc_h)
            acf = _d2(qf_s, dt, acc_h); acl = _d2(ql_s, dt, acc_h)
        qdf = np.array([float(r.get("target_front_back_deg") or 0.0) for r in rows])
        qdl = np.array([float(r.get("target_left_right_deg") or 0.0) for r in rows])
        # 传感器对齐（最近邻）
        gi = np.clip(np.searchsorted(imu[0], ts), 0, len(imu[0]) - 1) if imu else None
        gfi = np.clip(np.searchsorted(fo[0], ts), 0, len(fo[0]) - 1) if fo else None
        hi = max(horizon, 1)
        for i in range(len(rows) - hi):
            # use_u_hist=False：历史里【不放 u】——否则 h 已含 u_{t-1}≈u_t，头部 u_t 输入被架空
            # （实测：打乱 u_t 后留出 RMSE 不变 → ∂F/∂u_t≈0 → 反解无界。见 validate_forward_inverse）
            u_prev = ((uf[i - 1], ul[i - 1]) if i > 0 else (uf[i], ul[i])) if use_u_hist else (0.0, 0.0)
            x = [qf[i], ql[i], vf[i], vl[i], u_prev[0], u_prev[1]]
            x += list(imu[1][gi[i]]) if imu else [0.0] * 6
            x += list(fo[1][gfi[i]]) if fo else [0.0] * 4
            X.append(x)
            if horizon > 0:
                # 跨拍视界：目标 = Δq̇ over H 拍 —— ∂(Δq̇)/∂u 在速度尺度（~τ）良态，
                # 而单拍 ∂q̈/∂u 实测≈0（伺服+传动 10ms 内几乎不产生加速度）→ 反解不可能。
                Y.append([vf[i + horizon] - vf[i], vl[i + horizon] - vl[i]])
            else:
                Y.append([af[i], al[i]])      # 目标 = 加速度（单拍）
            S.append(sid)
            if with_aux:
                AUX["q"].append([qf[i], ql[i]]); AUX["q_d"].append([qdf[i], qdl[i]])
                AUX["u"].append([uf[i], ul[i]])
                AUX["qddot_causal"].append([acf[i], acl[i]])
                AUX["sid"].append(sid); AUX["t"].append(st[i])
    if not X:
        return None
    if with_aux:
        return (np.asarray(X), np.asarray(Y), np.asarray(S),
                {k: np.asarray(v) for k, v in AUX.items()})
    return np.asarray(X), np.asarray(Y), np.asarray(S)


def _rows_cols(path: Path, cols):
    rows = _rows(path)
    if not rows:
        return None, None
    return rows, cols


def build_windows(X, Y, S, L: int, stride: int = 1, y_shift: int = 1):
    """按 segment 切滑窗 → (W (m,L,16), Yt (m,2), U (m,2), sid (m,))。

    U = 窗口末拍的【当前动作】u_t（供头部反解）；窗口 x 内含的是 u_{t-1}（历史）。
    """
    Ws, Ys, Us, Ss, Is = [], [], [], [], []
    for sid in np.unique(S):
        idx = np.where(S == sid)[0]
        x, y = X[idx], Y[idx]
        if len(idx) < L + 2:
            continue
        for k in range(L - 1, len(idx) - 1, stride):
            Ws.append(x[k - L + 1:k + 1])
            Ys.append(y[k + y_shift])                 # q̈ 模式=y[k+1]；视界模式=y[k]
            Us.append([x[k + 1, 4], x[k + 1, 5]])     # u_t（已是下一拍的 u_prev）
            Ss.append(sid)
            Is.append(idx[k + 1])                 # 目标拍在会话内 X 的全局行号
    if not Ws:
        return None
    return (np.asarray(Ws), np.asarray(Ys), np.asarray(Us), np.asarray(Ss),
            np.asarray(Is))


# ---------------------------------------------------------------------------
# 训练（torch；导出 numpy 运行时）
# ---------------------------------------------------------------------------

def train(W, Yt, U, hidden=64, epochs=60, lr=2e-3, batch=256, seed=0, val_frac=0.15,
          n_feat=None):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n = len(W)
    perm = np.random.default_rng(seed).permutation(n)
    nv = max(256, int(val_frac * n))
    va, tr = perm[:nv], perm[nv:]

    nf = int(n_feat or W.shape[2])
    xm, xs = W[tr].reshape(-1, nf).mean(0), W[tr].reshape(-1, nf).std(0) + 1e-8
    um, us = U[tr].mean(0), U[tr].std(0) + 1e-8
    ym, ys = Yt[tr].mean(0), Yt[tr].std(0) + 1e-8

    def prep(idx):
        w = torch.tensor((W[idx] - xm) / xs, dtype=torch.float32)
        u = torch.tensor((U[idx] - um) / us, dtype=torch.float32)
        y = torch.tensor((Yt[idx] - ym) / ys, dtype=torch.float32)
        return w, u, y

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(nf, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + 2, 64), nn.Tanh(), nn.Linear(64, 2))

        def forward(self, w, u):
            _, h = self.gru(w)          # h: (1, B, hidden)
            return self.head(torch.cat([h[-1], u], dim=1))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    wv, uv, yv = prep(va)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        idx = np.random.default_rng(seed + ep).permutation(len(tr))
        for b in range(0, len(idx), batch):
            sel = idx[b:b + batch]
            if len(sel) < 8:
                continue
            wb, ub, yb = prep(tr[sel])
            opt.zero_grad()
            loss = lossf(net(wb, ub), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(wv, uv), yv).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep+1:3d}  val MSE(标准) {vl:.5f}")
    if best_state:
        net.load_state_dict(best_state)
    return net, dict(xm=xm, xs=xs, um=um, us=us, ym=ym, ys=ys), (va, tr)


def export(net, norm, hidden, u_hist=True, n_feat=None, features=None) -> dict:
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    out = {
        "in_mean": norm["xm"].tolist(), "in_std": norm["xs"].tolist(),
        "u_mean": norm["um"].tolist(), "u_std": norm["us"].tolist(),
        "out_mean": norm["ym"].tolist(), "out_std": norm["ys"].tolist(),
        "gru_W_ih": sd["gru.weight_ih_l0"].tolist(), "gru_W_hh": sd["gru.weight_hh_l0"].tolist(),
        "gru_b_ih": sd["gru.bias_ih_l0"].tolist(), "gru_b_hh": sd["gru.bias_hh_l0"].tolist(),
        "head_W0": sd["head.0.weight"].tolist(), "head_b0": sd["head.0.bias"].tolist(),
        "head_W1": sd["head.2.weight"].tolist(), "head_b1": sd["head.2.bias"].tolist(),
        "hidden": hidden, "n_feat": int(n_feat or N_FEAT),
        "features": np.asarray(features or FEATURES),
        "u_hist": np.asarray(bool(u_hist)),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="反解 F 模式：GRU 前向时序模型训练")
    ap.add_argument("--session", nargs="+", required=True, help="会话目录（≥3 供 LOSO）")
    ap.add_argument("--out", default="configs/forward_seq_v1.json")
    ap.add_argument("--seq-len", type=int, default=20, help="历史窗长（拍，100Hz → 20=0.2s）")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smooth", type=int, default=25, help="动捕平滑窗（拍）；压微分噪声")
    ap.add_argument("--acc-h", type=int, default=15, help="加速度中心差分半宽（拍）")
    ap.add_argument("--stride", type=int, default=5, help="滑窗步长（降冗余+算力）")
    ap.add_argument("--vel-mode", choices=["central", "poly"], default="poly",
                    help="q̇ 特征估计器：poly=因果零滞后（可上机）；central=中心差分（非因果，仅对照）")
    ap.add_argument("--vel-w", type=int, default=31, help="poly 估计器窗长（拍）")
    ap.add_argument("--horizon", type=int, default=0,
                    help="跨拍视界 H（拍）：目标 = q̇(t+H)−q̇(t)。0=单拍 q̈ 模式")
    ap.add_argument("--no-u-hist", action="store_true",
                    help="历史特征不含 u（逼头部 u_t 承担全部 u 依赖，反解必需）")
    ap.add_argument("--holdout", type=int, default=None, help="留出第几个会话（缺省=最后一个）")
    ap.add_argument("--train-all", action="store_true", help="生产模式：全部数据训练 + 随机验证")
    args = ap.parse_args()

    parts = []
    for s in args.session:
        p = Path(s)
        d = p if p.is_dir() else p.parent
        r = load_session(d, args.smooth, args.acc_h, args.vel_mode, args.vel_w,
                         use_u_hist=not args.no_u_hist, horizon=args.horizon)
        if r is None:
            print(f"跳过: {d.name}", file=sys.stderr)
            continue
        parts.append((d.name, r))
        print(f"{d.name}: {len(r[0])} 拍")
    if len(parts) < 2:
        return 1

    # 窗
    wins = []
    for name, (X, Y, S) in parts:
        w = build_windows(X, Y, S, args.seq_len, args.stride,
                          y_shift=0 if args.horizon > 0 else 1)
        if w:
            wins.append((name, w))
            print(f"  {name}: {len(w[0])} 窗")

    if args.train_all:
        W = np.concatenate([w[0] for _, w in wins])
        Yt = np.concatenate([w[1] for _, w in wins])
        U = np.concatenate([w[2] for _, w in wins])
        print(f"\n【train-all】{len(W)} 窗")
        net, norm, _ = train(W, Yt, U, args.hidden, args.epochs)
        # 全量评估（样本内，仅参考）
        import torch
        with torch.no_grad():
            w = torch.tensor((W - norm["xm"]) / norm["xs"], dtype=torch.float32)
            u = torch.tensor((U - norm["um"]) / norm["us"], dtype=torch.float32)
            pred = net(w, u).numpy() * norm["ys"] + norm["ym"]
        for a, ax in enumerate(["fb", "lr"]):
            r0 = float(np.sqrt(np.mean(Yt[:, a] ** 2)))
            r1 = float(np.sqrt(np.mean((Yt[:, a] - pred[:, a]) ** 2)))
            print(f"  [{ax}] q̈ RMSE {r0:.2f} → {r1:.2f} ({(r1/r0-1)*100:+.1f}%)")
    else:
        h = args.holdout if args.holdout is not None else len(wins) - 1
        W = np.concatenate([w[0] for i, (_, w) in enumerate(wins) if i != h])
        Yt = np.concatenate([w[1] for i, (_, w) in enumerate(wins) if i != h])
        U = np.concatenate([w[2] for i, (_, w) in enumerate(wins) if i != h])
        Wh, Yh, Uh = wins[h][1][:3]
        print(f"\n训练 {len(W)} 窗 / 留出 {wins[h][0]} {len(Wh)} 窗")
        net, norm, _ = train(W, Yt, U, args.hidden, args.epochs)
        import torch
        with torch.no_grad():
            w = torch.tensor((Wh - norm["xm"]) / norm["xs"], dtype=torch.float32)
            u = torch.tensor((Uh - norm["um"]) / norm["us"], dtype=torch.float32)
            pred = net(w, u).numpy() * norm["ys"] + norm["ym"]
        print(f"\n=== 留出会话 {wins[h][0]} ===")
        for a, ax in enumerate(["fb", "lr"]):
            r0 = float(np.sqrt(np.mean(Yh[:, a] ** 2)))
            r1 = float(np.sqrt(np.mean((Yh[:, a] - pred[:, a]) ** 2)))
            print(f"  [{ax}] q̈ RMSE {r0:.2f} → {r1:.2f}（{(r1/r0-1)*100:+.1f}%）")

    nn_dict = export(net, norm, args.hidden, u_hist=not args.no_u_hist)
    out = Path(args.out)
    npz = out.with_suffix(".npz")
    arrs = {}
    for k, v in nn_dict.items():
        arrs[k] = np.asarray(v) if isinstance(v, (list, str)) else np.asarray(v)
    np.savez_compressed(npz, **arrs)
    data = {"forward_seq_npz": npz.name,
            "_note": (f"反解 F 模式：GRU 时序前向模型 | 输入 {N_FEAT} 维（状态+IMU+张力历史）"
                      f"窗长 {args.seq_len} 拍 | hidden {args.hidden} | 预测 q̈ | "
                      f"q̇特征 {args.vel_mode}(w={args.vel_w}) | 历史含u={not args.no_u_hist} | "
                      f"标签 中心差分(smooth={args.smooth},h={args.acc_h}) | "
                      f"训练 {'train-all' if args.train_all else '留出'} | 权重在 {npz.name}")}
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写: {out}（{out.stat().st_size} B）+ {npz.name}（{npz.stat().st_size/1024:.1f} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
