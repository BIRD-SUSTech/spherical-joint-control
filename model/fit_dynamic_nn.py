"""§8.4 D1：动态残差 MLP 训练 + 导出（架构1，残差式）。

    u_ff = g_static(q_d) + f_θ(q_d, q̇_d)

监督信号（闭环收敛段）：`u_总 ≈ g_static(q_d) + 残差` → 目标残差 `r = u_总 − g_static(q_d)`。
- 静态基座 `g_static` **固定不训练**（来自 --base-config 的 gain_poly/gain_cross）；
- 网络只学修正量 → **缺省/零权重即精确退回当前稳态前馈**（不产生负优化）；
- 输入只用**解析参考量** `q_d, q̇_d`（轨迹生成器给，零噪声），保持纯前馈、无闭环风险。

留出：按【轨迹/会话】留出（不是按时间），考泛化到新轨迹。
导出：`ControllerConfig.dynamic_nn`（numpy 前向，运行时**无 torch 依赖**）。

用法：
    python -m model.fit_dynamic_nn --session <A> <B> <C> \
        --base-config configs/static_feedforward_controller.json \
        --out configs/dynamic_nn_v1.json --epochs 300 --threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from control.controller_config import ControllerConfig

FEATURES_V1 = ["q_d_fb", "q_d_lr", "qdot_d_fb", "qdot_d_lr"]
FEATURES_V2 = FEATURES_V1 + ["qddot_d_fb", "qddot_d_lr"]
FEATURES_IMU = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
FEATURES_FORCE = ["ch1", "ch2", "ch3", "ch4"]
# 特征名 → CSV 实际列名（单一事实源；运行时按特征名取值，故此处只影响训练取数）
SENSOR_COLS = {"gyro_x": "gyro_x_dps", "gyro_y": "gyro_y_dps", "gyro_z": "gyro_z_dps",
               "acc_x": "ax_no_g_mps2", "acc_y": "ay_no_g_mps2", "acc_z": "az_no_g_mps2",
               "ch1": "ch1", "ch2": "ch2", "ch3": "ch3", "ch4": "ch4"}
FEATURES = FEATURES_V1   # 兼容旧引用（默认 v1）


def feature_names(use_qddot: bool, use_imu: bool = False, use_force: bool = False) -> list:
    """特征顺序的**单一事实源**：训练 / 导出 / 运行时三处必须一致。"""
    names = list(FEATURES_V2 if use_qddot else FEATURES_V1)
    if use_imu:
        names += FEATURES_IMU
    if use_force:
        names += FEATURES_FORCE
    return names


def _load_sensor(session_dir: Path, filename: str, cols: list):
    """读一个传感器 CSV → (ts_ms 排序后, vals (n,len(cols)))。缺文件→None。"""
    p = session_dir / filename
    if not p.exists():
        return None
    ts, vals = [], []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ts.append(int(row["pc_receive_unix_time_ms"]))
                vals.append([float(row[c]) for c in cols])
            except (KeyError, ValueError):
                continue
    if not ts:
        return None
    ts = np.asarray(ts)
    vals = np.asarray(vals)
    o = np.argsort(ts)
    return ts[o], vals[o]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _poly(coeffs, x):
    return sum(c * x ** k for k, c in enumerate(coeffs))


def _poly_no_const(coeffs, x):
    return sum(c * x ** (k + 1) for k, c in enumerate(coeffs))


def base_static_u(cfg: ControllerConfig, q_fb: float, q_lr: float,
                  v_fb: float = 0.0, v_lr: float = 0.0) -> tuple[float, float]:
    """基座前馈：g(q_d + τ·q̇_d)（含 velocity_lead 相位超前）。

    **不含** dynamic_nn，也不含迟滞——NN 学的是"基座之后的残余"。
    """
    qe_fb, qe_lr = q_fb, q_lr
    if cfg.velocity_lead is not None:
        qe_fb += cfg.velocity_lead.get("fb", 0.0) * v_fb
        qe_lr += cfg.velocity_lead.get("lr", 0.0) * v_lr
    if cfg.gain_poly is not None:
        u_fb = _poly(cfg.gain_poly["fb"], qe_fb)
        u_lr = _poly(cfg.gain_poly["lr"], qe_lr)
    elif cfg.direction_gains is not None:
        g = cfg.direction_gains
        u_fb = qe_fb / (g["fb"]["pos"] if qe_fb >= 0 else g["fb"]["neg"])
        u_lr = qe_lr / (g["lr"]["pos"] if qe_lr >= 0 else g["lr"]["neg"])
    else:
        u_fb = u_lr = 0.0
    if cfg.gain_cross is not None:
        u_fb += _poly_no_const(cfg.gain_cross["fb"], q_lr)
        u_lr += _poly_no_const(cfg.gain_cross["lr"], q_fb)
    return u_fb, u_lr


def load_session_features(csv_path: Path, cfg: ControllerConfig, threshold: float,
                          include_all: bool = False, use_qddot: bool = False,
                          use_imu: bool = False, use_force: bool = False):
    """读一个会话 → (features (n,k), residual (n,2), seg (n,))。

    use_imu/use_force=True 时，从同目录 imu_data.csv / force_data.csv **按时间戳最近邻**
    对齐后追加 6/4 维传感器特征（实测状态）。注意：这会把本模型从"纯前馈"变成
    **含实测状态反馈的项**，需单独稳定性分析 + 非指令频率检查（见 D4a 否决报告）。

    只取收敛段（|current−target| < threshold 两轴都满足）：此时 `u_总 ≈ 目标状态所需 u`，
    残差 r = u_总 − g_static(q_d) 即"静态前馈没吃掉的动态量"。
    include_all=True 时不筛收敛段（供对照）。
    """
    # 先收集【全部】行并按时间排序：参考速度必须在【连续】时间序列上求。
    # （若先滤掉非收敛样本再求梯度，序列出现空洞却仍按均匀 dt 差分 → 跨洞处产生
    #  上千 °/s 的伪峰，实测可达 1967°/s，而物理上限仅约 63°/s。）
    rows_all: dict[int, list] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("segment_id", "")
            if sid in ("", "-1", None):
                continue
            try:
                ts = float(row["t_s"])
                tf = float(row["target_front_back_deg"])
                tl = float(row["target_left_right_deg"])
                cf = float(row["current_front_back_deg"])
                cl = float(row["current_left_right_deg"])
                uf = float(row["servo_front_back_offset"])
                ul = float(row["servo_left_right_offset"])
            except (KeyError, ValueError):
                continue
            rows_all.setdefault(int(sid), []).append(
                (ts, tf, tl, cf, cl, uf, ul, int(row["pc_receive_unix_time_ms"])))

    sdir = csv_path.parent
    imu = _load_sensor(sdir, "imu_data.csv",
                       [SENSOR_COLS[c] for c in FEATURES_IMU]) if use_imu else None
    fo = _load_sensor(sdir, "force_data.csv",
                      [SENSOR_COLS[c] for c in FEATURES_FORCE]) if use_force else None
    if use_imu and imu is None:
        print(f"警告：{sdir.name} 无 imu_data.csv → IMU 特征置 0", file=sys.stderr)
    if use_force and fo is None:
        print(f"警告：{sdir.name} 无 force_data.csv → 力特征置 0", file=sys.stderr)

    feats, res, seg_ids = [], [], []
    for sid, rows in rows_all.items():
        if len(rows) < 3:
            continue
        rows.sort(key=lambda r: r[0])
        ts = np.array([r[0] for r in rows])
        tf = np.array([r[1] for r in rows])
        tl = np.array([r[2] for r in rows])
        cf = np.array([r[3] for r in rows])
        cl = np.array([r[4] for r in rows])
        uf = np.array([r[5] for r in rows])
        ul = np.array([r[6] for r in rows])
        tsms = np.array([r[7] for r in rows])
        # 传感器最近邻对齐（与 servo 时钟）
        if imu is not None:
            ii = np.clip(np.searchsorted(imu[0], tsms, side="left"), 0, len(imu[0]) - 1)
            ii = np.where((ii > 0) & (np.abs(imu[0][np.maximum(ii - 1, 0)] - tsms)
                                      < np.abs(imu[0][ii] - tsms)), ii - 1, ii)
            imu_v = imu[1][ii]
        if fo is not None:
            fi = np.clip(np.searchsorted(fo[0], tsms, side="left"), 0, len(fo[0]) - 1)
            fi = np.where((fi > 0) & (np.abs(fo[0][np.maximum(fi - 1, 0)] - tsms)
                                      < np.abs(fo[0][fi] - tsms)), fi - 1, fi)
            fo_v = fo[1][fi]
        # 参考速度：在【全序列】上差分（目标平滑 → 干净，且不是实测差分）
        dt = np.median(np.diff(ts)) if len(ts) > 1 else 0.01
        vf = np.gradient(tf, dt)
        vl = np.gradient(tl, dt)
        # 加速度同样在【全序列】上求（二阶差分；目标解析平滑 → 干净）
        af = np.gradient(vf, dt)
        al = np.gradient(vl, dt)
        for i in range(len(ts)):
            if not include_all and (abs(cf[i] - tf[i]) >= threshold
                                    or abs(cl[i] - tl[i]) >= threshold):
                continue
            bf, bl = base_static_u(cfg, tf[i], tl[i], vf[i], vl[i])
            row = [tf[i], tl[i], vf[i], vl[i]] + ([af[i], al[i]] if use_qddot else [])
            if imu is not None:
                row += list(imu_v[i])
            if fo is not None:
                row += list(fo_v[i])
            feats.append(row)
            res.append([uf[i] - bf, ul[i] - bl])
            seg_ids.append(sid)
    return np.array(feats), np.array(res), np.array(seg_ids)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_mlp(X_tr, R_tr, X_va, R_va, hidden=32, depth=2, epochs=300, lr=1e-3, seed=0):
    """训练 MLP，返回 (model, 归一化参数, 训练史)。torch 惰性导入。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    in_mean, in_std = X_tr.mean(0), X_tr.std(0) + 1e-8
    out_mean, out_std = R_tr.mean(0), R_tr.std(0) + 1e-8
    Xn = torch.tensor((X_tr - in_mean) / in_std, dtype=torch.float32)
    Rn = torch.tensor((R_tr - out_mean) / out_std, dtype=torch.float32)
    Xv = torch.tensor((X_va - in_mean) / in_std, dtype=torch.float32)
    Rv = torch.tensor((R_va - out_mean) / out_std, dtype=torch.float32)

    layers, d = [], X_tr.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers += [nn.Linear(d, R_tr.shape[1])]
    net = nn.Sequential(*layers)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best, best_state, hist = float("inf"), None, []
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xn), Rn)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xv), Rv).item()
        hist.append((loss.item(), vl))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, (in_mean, in_std, out_mean, out_std), hist


def export_nn(net, norm, clip: float, features=None) -> dict:
    """torch 模型 → dict（层权重 + 归一化 + 限幅），供运行时 numpy 前向。

    **关键**：把输出反归一化（out_mean/out_std）**折叠进最后一层**，导出 out_mean=0/out_std=1。
    这样"权重与偏置全部置零"时网络输出**恰好为 0** → 精确退回纯稳态前馈（§8.4.2 零退回保证）。
    若保留 out_mean 在反归一化里，置零会得到常数 out_mean ≠ 0，保证就失效。
    """
    in_mean, in_std, out_mean, out_std = norm
    features = None
    layers = []
    for m in net:
        if hasattr(m, "weight"):  # nn.Linear
            layers.append({"W": m.weight.detach().numpy().tolist(),
                           "b": m.bias.detach().numpy().tolist()})
    last = layers[-1]
    W_last = np.asarray(last["W"])            # (n_out, n_hidden)
    b_last = np.asarray(last["b"])            # (n_out,)
    last["W"] = (W_last * out_std[:, None]).tolist()
    last["b"] = (b_last * out_std + out_mean).tolist()
    return {
        "in_mean": in_mean.tolist(), "in_std": in_std.tolist(),
        "out_mean": [0.0] * len(out_mean), "out_std": [1.0] * len(out_std),
        "layers": layers, "act": "tanh", "clip": clip,
        "n_in": int(len(in_mean)),                       # 4=v1 / 6=v2(+q̈) / +6(+IMU) / +4(+力)
        "features": [str(x) for x in (features if features else
                                      (FEATURES_V2 if len(in_mean) >= 6 else FEATURES_V1))],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="§8.4 D1 动态残差 MLP 训练 + 导出")
    ap.add_argument("--session", nargs="+", required=True, help="会话目录或 servo_data.csv（多个）")
    ap.add_argument("--base-config", required=True, help="静态前馈配置（提供 g_static）")
    ap.add_argument("--out", default="configs/dynamic_nn_v1.json", help="输出配置")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="收敛阈值（度）：|current−target| 小于它才算监督样本")
    ap.add_argument("--holdout-session", type=int, default=None,
                    help="留出第几个会话（缺省=最后一个），考泛化到新轨迹")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-all", action="store_true",
                    help="生产模式：用全部会话训练（随机 15%% 做早停验证），不浪费一个会话做留出")
    ap.add_argument("--imu", action="store_true",
                    help="输入加 IMU 陀螺+加计（6 维，实测）——注意：会变成含实测状态的反馈项")
    ap.add_argument("--force", action="store_true",
                    help="输入加张力 ch1–ch4（4 维，实测）——注意：同上")
    ap.add_argument("--qddot", action="store_true",
                    help="输入加 q̈_d（v2；D0 实证：LOSO 留出 R² +0.02~+0.10，远超双控制组）")
    ap.add_argument("--clip", type=float, default=None,
                    help="残差输出限幅（offset）；缺省=3×训练残差最大值")
    args = ap.parse_args()

    cfg = ControllerConfig.load(args.base_config)
    if cfg.gain_poly is None and cfg.direction_gains is None:
        print("base-config 缺 gain_poly/direction_gains，无法定义 g_static", file=sys.stderr)
        return 1
    if cfg.dynamic_nn is not None:
        print("注意：base-config 已含 dynamic_nn，本脚本只用其静态部分作为基座", file=sys.stderr)

    paths = []
    for s in args.session:
        p = Path(s)
        csv_path = p / "servo_data.csv" if p.is_dir() else p
        if not csv_path.exists():
            print(f"文件不存在: {csv_path}", file=sys.stderr)
            return 1
        paths.append(csv_path)

    per_session = []
    for p in paths:
        X, R, seg = load_session_features(p, cfg, args.threshold, use_qddot=args.qddot,
                                         use_imu=args.imu, use_force=args.force)
        print(f"{p.parent.name}: 收敛样本 {len(X)}")
        if len(X):
            per_session.append((X, R))
    if len(per_session) < 2:
        print("有效会话不足 2 个（需留出泛化）", file=sys.stderr)
        return 1

    if args.train_all:
        X_all = np.concatenate([s[0] for s in per_session])
        R_all = np.concatenate([s[1] for s in per_session])
        n_val = max(200, int(0.15 * len(X_all)))
        idx = np.random.default_rng(args.seed).permutation(len(X_all))
        va, tr = idx[:n_val], idx[n_val:]
        X_tr, R_tr, X_va, R_va = X_all[tr], R_all[tr], X_all[va], R_all[va]
        print(f"\n【train-all】训练 {len(X_tr)} / 内部验证 {len(X_va)}（同分布随机切分，供早停）")
    else:
        hidx = args.holdout_session if args.holdout_session is not None else len(per_session) - 1
        X_va, R_va = per_session[hidx]
        X_tr = np.concatenate([s[0] for i, s in enumerate(per_session) if i != hidx])
        R_tr = np.concatenate([s[1] for i, s in enumerate(per_session) if i != hidx])
        print(f"\n训练 {len(X_tr)} 样本 / 留出会话 #{hidx} {len(X_va)} 样本")

    # 基线：残差全零（=当前稳态前馈）在留出集上的 RMSE —— 改进必须低于它
    rmse0 = float(np.sqrt(np.mean(R_va ** 2)))
    print(f"留出集残差 RMSE（零修正基线）: fb={rmse0:.2f}" if R_va.shape[1] == 1 else
          f"留出集残差 RMSE（零修正基线）: fb="
          f"{np.sqrt(np.mean(R_va[:,0]**2)):.2f} lr={np.sqrt(np.mean(R_va[:,1]**2)):.2f}")

    net, norm, hist = train_mlp(X_tr, R_tr, X_va, R_va, args.hidden, args.depth,
                                args.epochs, args.lr, args.seed)

    # 留出评估（离线，仅作拟合优度参考；判据仍以实机 A/B 为准）
    import torch
    with torch.no_grad():
        in_mean, in_std, out_mean, out_std = norm
        Xv = torch.tensor((X_va - in_mean) / in_std, dtype=torch.float32)
        pred = net(Xv).numpy() * out_std + out_mean
    for a, ax in enumerate(["fb", "lr"]):
        r0 = float(np.sqrt(np.mean(R_va[:, a] ** 2)))
        r1 = float(np.sqrt(np.mean((R_va[:, a] - pred[:, a]) ** 2)))
        r2 = 1.0 - np.var(R_va[:, a] - pred[:, a]) / np.var(R_va[:, a])
        print(f"  [{ax}] 留出残差 RMSE {r0:.2f} → {r1:.2f}（{100*(r1/r0-1):+.1f}%）  R²={r2:.3f}")

    clip = args.clip if args.clip is not None else float(3.0 * np.abs(R_tr).max())
    feat_names = feature_names(args.qddot, args.imu, args.force)
    net_dict = export_nn(net, norm, clip, features=feat_names)
    print(f"特征（{len(feat_names)} 维）: {', '.join(feat_names)}")

    from control.controller_config import save_nn_npz
    out = Path(args.out)
    npz = out.with_suffix(".npz")
    save_nn_npz(npz, net_dict)
    data = json.loads(Path(args.base_config).read_text(encoding="utf-8-sig"))
    data.pop("dynamic_nn", None)
    data.pop("dynamic_nn_npz", None)
    data["dynamic_nn_npz"] = npz.name          # 与配置文件同目录，用相对名（可移植）
    base_has_lead = cfg.velocity_lead is not None
    base_desc = "g(q_d+τ·q̇_d)" if base_has_lead else "g_static(q_d)"
    data["_note"] = (
        f"§8.4 动态残差 MLP（**残差式**）：u_ff = {base_desc}[基座固定] + f_θ(q_d,q̇_d"
        f"{',q̈_d' if args.qddot else ''}) | "
        f"训练 {len(paths)} 会话（{'train-all' if args.train_all else '留出'}）"
        f"threshold={args.threshold}° hidden={args.hidden} depth={args.depth} epochs={args.epochs} | "
        f"n_in={net_dict['n_in']} clip={clip:.0f} | 基座配置={Path(args.base_config).name} | "
        f"权重在 {out.with_suffix('.npz').name}（JSON 不内联） | 权重置零=精确退回基座")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写: {out}（JSON 仅 {out.stat().st_size} B，权重在 {npz.name}）")
    print(f"      权重 npz: {npz}（{npz.stat().st_size/1024:.1f} KB）  clip={clip:.0f} offset")
    print("⚠️ 离线指标只衡量拟合优度，不作判据——merge 与否由实机 A/B 决定（§8.4.6）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
