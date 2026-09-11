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
FEATURES = FEATURES_V1   # 兼容旧引用（默认 v1）


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _poly(coeffs, x):
    return sum(c * x ** k for k, c in enumerate(coeffs))


def _poly_no_const(coeffs, x):
    return sum(c * x ** (k + 1) for k, c in enumerate(coeffs))


def base_static_u(cfg: ControllerConfig, q_fb: float, q_lr: float) -> tuple[float, float]:
    """静态基座 g_static（**不含** dynamic_nn，也不含迟滞）。"""
    if cfg.gain_poly is not None:
        u_fb = _poly(cfg.gain_poly["fb"], q_fb)
        u_lr = _poly(cfg.gain_poly["lr"], q_lr)
    elif cfg.direction_gains is not None:
        g = cfg.direction_gains
        u_fb = q_fb / (g["fb"]["pos"] if q_fb >= 0 else g["fb"]["neg"])
        u_lr = q_lr / (g["lr"]["pos"] if q_lr >= 0 else g["lr"]["neg"])
    else:
        u_fb = u_lr = 0.0
    if cfg.gain_cross is not None:
        u_fb += _poly_no_const(cfg.gain_cross["fb"], q_lr)
        u_lr += _poly_no_const(cfg.gain_cross["lr"], q_fb)
    return u_fb, u_lr


def load_session_features(csv_path: Path, cfg: ControllerConfig, threshold: float,
                          include_all: bool = False, use_qddot: bool = False):
    """读一个会话 → (features (n,4), residual (n,2), seg (n,))。

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
            rows_all.setdefault(int(sid), []).append((ts, tf, tl, cf, cl, uf, ul))

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
            bf, bl = base_static_u(cfg, tf[i], tl[i])
            if use_qddot:
                feats.append([tf[i], tl[i], vf[i], vl[i], af[i], al[i]])
            else:
                feats.append([tf[i], tl[i], vf[i], vl[i]])
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


def export_nn(net, norm, clip: float) -> dict:
    """torch 模型 → dict（层权重 + 归一化 + 限幅），供运行时 numpy 前向。

    **关键**：把输出反归一化（out_mean/out_std）**折叠进最后一层**，导出 out_mean=0/out_std=1。
    这样"权重与偏置全部置零"时网络输出**恰好为 0** → 精确退回纯稳态前馈（§8.4.2 零退回保证）。
    若保留 out_mean 在反归一化里，置零会得到常数 out_mean ≠ 0，保证就失效。
    """
    in_mean, in_std, out_mean, out_std = norm
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
        "n_in": int(len(in_mean)),                       # 4=v1(无q̈) / 6=v2(含q̈)
        "features": FEATURES_V2 if len(in_mean) >= 6 else FEATURES_V1,
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
        X, R, seg = load_session_features(p, cfg, args.threshold, use_qddot=args.qddot)
        print(f"{p.parent.name}: 收敛样本 {len(X)}")
        if len(X):
            per_session.append((X, R))
    if len(per_session) < 2:
        print("有效会话不足 2 个（需留出泛化）", file=sys.stderr)
        return 1

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
    net_dict = export_nn(net, norm, clip)

    data = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    data["dynamic_nn"] = net_dict
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写: {out}（clip={clip:.0f} offset；含 base-config 的全部静态项）")
    print("⚠️ 离线指标只衡量拟合优度，不作判据——merge 与否由实机 A/B 决定（§8.4.6）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
