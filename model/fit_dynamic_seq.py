"""§8.4 **时序动态前馈模型（论文主模型）**：GRU(全部传感器历史) → Δu。

## 架构（残差式，零回归保证）

    u_ff = g_static(q_d)  [基座，固定]           ← 静态逆映射（既有成果）
         + clip( Δu_θ , ±cap )                  ← 时序模型修正；Δu=0 逐位退回基座

    Δu_θ = GRU( x_{t-L+1..t} ) → 线性头           L=10 拍（0.1s @100Hz）

    x 每拍 22 维：
        实测状态  q(2) q̇(2)
        期望轨迹  q_d(2) q̇_d(2) q̈_d(2)           ← 外生（控制器给定）
        IMU       陀螺(3) 加计(3)
        张力      ch1–ch4
        上拍动作  u_prev(2)

模型/训练/推理**全部走 torch**（`model/seq_model.py`），device 自动选 cuda。

## 监督信号（关键，沿用 v3/D1 成熟协议）

    标签 Δu = u_recorded(t) − u_base(q_d, q̇_d)，**只在收敛段取样本**（|q−q_d| < threshold）。

不筛收敛段会怎样（实测）：u_recorded 含 PID 反馈，标签就成了 PID 输出的确定性函数
（同拍特征线性回归 R² 即达 0.986），模型退化为"复刻 PID"而非"学被控对象需求"。
收敛段 PID≈0 → 标签≈该目标状态真实需要的动作量。

## 评价协议

逐会话 LOSO 留出（4 会话）+ 内部 15% 作早停验证。对照 = 零修正（纯静态基座）。
离线只做筛选；上机判定要求多轨迹一致（单轨迹已三次误判）。

用法：
    python -m model.fit_dynamic_seq --session <A> <B> <C> <D> --loso --train-all \
        --out configs/dynamic_seq_v4.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.seq_model import fit, save_model, pick_device            # noqa: E402

# ---------------------------------------------------------------------------
# 数据加载：四路传感器按 servo 时钟最近邻对齐 → 逐拍特征
# （自 model/fit_forward_seq.py 移入；反解 F 路线已否决，只保留此处需要的数据层）
# ---------------------------------------------------------------------------

_X_FEAT = 16          # load_session 内部布局：q(2) q̇(2) u_prev(2) gyro(3) acc(3) ch(4)

def _rows(path: Path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows_cols(path: Path, cols):
    rows = _rows(path)
    if not rows:
        return None, None
    return rows, cols


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

# ---------------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------------

FEATURES = ["q_fb", "q_lr", "qdot_fb", "qdot_lr",
            "qd_fb", "qd_lr", "qdotd_fb", "qdotd_lr", "qddotd_fb", "qddotd_lr",
            "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
            "ch1", "ch2", "ch3", "ch4", "u_prev_fb", "u_prev_lr"]
N_FEAT = len(FEATURES)


def build_dataset(session: Path, seq_len: int, stride: int, vel_w: int,
                  threshold: float = 1.0, base=None, dt: float = None):
    """→ (W (m,L,22), Y (m,2)=Δu, I, feats, u, qd, du)；窗口末拍为收敛段。"""
    r = load_session(session, 25, 15, "poly", vel_w, with_aux=True)
    if r is None:
        return None
    X, _, S, AUX = r
    n = len(X)
    q, qd, u = AUX["q"], AUX["q_d"], AUX["u"]
    # dt：**用实测采样间隔**（物理正确）。load_session 算实测 q̇ 时用的也是实测 dt，
    # 两侧必须同口径——否则 q̇/q̈ 特征静默漂移（实测差 ~1 offset）。
    if dt is None:
        dt = float(np.median(np.diff(AUX["t"])))
    return_dt = dt
    vd = np.column_stack([_poly_est(qd[:, 0], dt, vel_w)[1], _poly_est(qd[:, 1], dt, vel_w)[1]])
    ad = np.column_stack([_poly_est(qd[:, 0], dt, vel_w)[2], _poly_est(qd[:, 1], dt, vel_w)[2]])

    feats = np.zeros((n, N_FEAT))
    feats[:, 0:2] = q                     # 实测角
    feats[:, 2:4] = X[:, 2:4]             # 实测 q̇（load_session 已用因果核算好）
    feats[:, 4:6] = qd
    feats[:, 6:8] = vd
    feats[:, 8:10] = ad
    feats[:, 10:16] = X[:, 6:12]          # IMU 陀螺+加计
    feats[:, 16:20] = X[:, 12:16]         # 张力 ch1–4
    # 上拍动作 u_{t-1}：**因果可用**（发指令前只知道上一拍发了什么）。
    # 用 u[t] 是错的（未来信息）——训练/运行时会静默错位，实测该列差 4~9 offset。
    feats[1:, 20:22] = u[:-1]
    feats[0, 20:22] = u[0]

    # 基座动作（未整形：slew 是有状态限速，会污染标签）
    if base is not None:
        ub = np.array([base.feedforward(qd[i, 0], qd[i, 1], vd[i, 0], vd[i, 1])
                       for i in range(n)])
    else:
        ub = np.zeros_like(u)
    du = u - ub
    conv = (np.abs(q[:, 0] - qd[:, 0]) < threshold) & (np.abs(q[:, 1] - qd[:, 1]) < threshold)

    Ws, Ys, Ss, Is = [], [], [], []
    for sid in np.unique(S):
        idx = np.where(S == sid)[0]
        if len(idx) < seq_len + 2:
            continue
        for k in range(seq_len - 1, len(idx), stride):
            t = idx[k]
            if not conv[t]:                # 只在收敛段监督（末拍）
                continue
            Ws.append(feats[t - seq_len + 1:t + 1])
            Ys.append(du[t])
            Ss.append(sid)
            Is.append(t)
    if not Ws:
        return None
    return (np.asarray(Ws), np.asarray(Ys), np.asarray(Ss), np.asarray(Is), feats, u, qd,
            du, return_dt)


def _score(model, W, Y):
    """→ {ax: (零修正基线 RMSE, 模型 RMSE, R²)}"""
    pred = model.predict_np(W)
    out = {}
    for a, ax in enumerate(["fb", "lr"]):
        b = float(np.sqrt(np.mean(Y[:, a] ** 2)))
        m = float(np.sqrt(np.mean((Y[:, a] - pred[:, a]) ** 2)))
        r2 = 1.0 - np.var(Y[:, a] - pred[:, a]) / max(float(np.var(Y[:, a])), 1e-12)
        out[ax] = (b, m, r2)
    return out


def _split(W, Y, seed=0, frac=0.85):
    """训练折内部切 15% 作早停验证（不污染 LOSO 留出折）。"""
    n = int(frac * len(W))
    idx = np.random.default_rng(seed).permutation(len(W))
    return W[idx[:n]], Y[idx[:n]], W[idx[n:]], Y[idx[n:]]


def main() -> int:
    ap = argparse.ArgumentParser(description="§8.4 时序动态前馈模型（GRU，全部传感器）")
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--out", default="configs/dynamic_seq_v4.json")
    ap.add_argument("--seq-len", type=int, default=10, help="历史窗长（拍）；10=0.1s@100Hz")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--vel-w", type=int, default=31, help="因果 q̇ 估计器窗长（拍）")
    ap.add_argument("--threshold", type=float, default=1.0, help="收敛段阈值（度）")
    ap.add_argument("--base-config", default="configs/static_feedforward_slew20_controller.json")
    ap.add_argument("--device", default="auto", help="auto/cuda/cpu")
    ap.add_argument("--loso", action="store_true", help="逐会话留出（论文评价协议）")
    ap.add_argument("--train-all", action="store_true", help="全量训练并导出部署件")
    args = ap.parse_args()

    from control.controller_config import ControllerConfig
    base = ControllerConfig.load(args.base_config)
    base.slew_limit = None
    base.dynamic_nn = None
    print(f"设备 {pick_device(args.device)} | 基座 {args.base_config} | 收敛阈值 {args.threshold}° | "
          f"窗长 {args.seq_len} 拍 | 特征 {N_FEAT} 维")

    parts = []
    for sdir in args.session:
        pth = Path(sdir)
        d = pth if pth.is_dir() else pth.parent
        r = build_dataset(d, args.seq_len, args.stride, args.vel_w, args.threshold, base)
        if r is None:
            print(f"  跳过 {d.name}（无收敛窗）", file=sys.stderr)
            continue
        W, Y, S, I, feats, u, qd, du, dt_s = r
        parts.append({"name": d.name, "W": W, "Y": Y, "dt": dt_s})
        print(f"  {d.name}: 收敛窗 {len(W)} | dt={dt_s:.5f}s | Δu std fb {np.std(du[:,0]):.2f} "
              f"lr {np.std(du[:,1]):.2f} | 全段 u std {np.std(u[:,0]):.2f}/{np.std(u[:,1]):.2f}")
    if len(parts) < 2:
        print("有效会话不足 2 个", file=sys.stderr)
        return 1

    if args.loso:
        print(f"\n{'留出会话':<22} {'零修正基线 fb/lr':>19} {'时序模型 fb/lr':>20} {'改善':>15}")
        print("-" * 80)
        agg = []
        for h in range(len(parts)):
            Wtr = np.concatenate([p["W"] for i, p in enumerate(parts) if i != h])
            Ytr = np.concatenate([p["Y"] for i, p in enumerate(parts) if i != h])
            a, b_, c, d_ = _split(Wtr, Ytr, 0)
            model, _ = fit(a, b_, args.hidden, args.epochs, device=args.device,
                           val=(c, d_), verbose=False)
            ev = _score(model, parts[h]["W"], parts[h]["Y"])
            bf, mf, _ = ev["fb"]; bl, ml, _ = ev["lr"]
            agg.append((bf, mf, bl, ml))
            print(f"{parts[h]['name']:<22} {bf:7.2f} /{bl:6.2f} {mf:11.2f} /{ml:9.2f} "
                  f"{(mf/bf-1)*100:+6.1f}%/{(ml/bl-1)*100:+.1f}%")
        ab = np.mean([x[0] for x in agg]); am = np.mean([x[1] for x in agg])
        bl2 = np.mean([x[2] for x in agg]); ml2 = np.mean([x[3] for x in agg])
        print("-" * 80)
        print(f"{'平均（LOSO 4 折）':<22} {ab:7.2f} /{bl2:6.2f} {am:11.2f} /{ml2:9.2f} "
              f"{(am/ab-1)*100:+6.1f}%/{(ml2/bl2-1)*100:+.1f}%")

    if args.train_all:
        W = np.concatenate([p["W"] for p in parts])
        Y = np.concatenate([p["Y"] for p in parts])
        print(f"\n【train-all】{len(W)} 窗")
        a, b_, c, d_ = _split(W, Y, 0)
        model, _ = fit(a, b_, args.hidden, args.epochs, device=args.device,
                       val=(c, d_), verbose=True)
        ev = _score(model, W, Y)
        for ax in ("fb", "lr"):
            bb, mm, r2 = ev[ax]
            print(f"  [{ax}] Δu RMSE {bb:.2f} → {mm:.2f} ({(mm/bb-1)*100:+.1f}%)  R²={r2:.3f}")

        out = Path(args.out)
        pt = out.with_suffix(".pt")
        dt_used = float(np.median([p["dt"] for p in parts]))
        save_model(model, pt, features=FEATURES,
                   meta={"base_config": args.base_config, "threshold": args.threshold,
                         "seq_len": args.seq_len, "vel_w": args.vel_w, "dt": dt_used,
                         "sessions": [p["name"] for p in parts], "n_win": int(len(W))})
        merged = json.loads(Path(args.base_config).read_text(encoding="utf-8-sig"))
        merged["dynamic_seq"] = {"model": pt.name, "seq_len": args.seq_len,
                                 "vel_w": args.vel_w, "dt": round(dt_used, 5),
                                 "mode": "direct_u",
                                 "cap": 0.0, "device": "cpu",
                                 "base_config": str(args.base_config)}
        merged["_note"] = (
            f"§8.4 时序动态前馈（论文主模型）：u_ff = g_static(q_d) + clip(Δu_GRU, ±cap) | "
            f"GRU 历史窗 {args.seq_len} 拍 | 特征 {N_FEAT} 维（实测状态+期望轨迹+IMU+张力+上拍动作）| "
            f"hidden {args.hidden} | 监督 Δu=u_recorded−u_base，收敛段 |q−q_d|<{args.threshold}° | "
            f"dt={dt_used:.5f}s | cap=0 逐位退回静态基座 | 权重在 {pt.name}（torch.save）")
        out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写 {out} + {pt.name}（{pt.stat().st_size/1024:.1f} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
