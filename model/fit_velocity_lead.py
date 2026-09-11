"""速度前馈（相位超前）τ 拟合：u_ff = g(q_d + τ·q̇_d)。

**依据**（D0 干净数据方差分解，实验机实测 + 开发机独立复核）：
    标准化标签 r ~ [q̇, q̈]：q̇ 系数 **0.85~0.98**、仅 q̇ R² = 0.875~0.955；
    q̈ 系数 −0.04~−0.25、仅 q̈ R² 仅 0.02~0.30 → **主导动态项是速度型，不是加速度型**。
    且 c_q̇ ≈ g′(q)·τ，8 数据集/轴 τ 全落在 0.169~0.252 s（均值 ≈0.19 s）。

物理含义：静态逆映射 g(q_d) 结构上给不出相位超前，而闭环纯 PID 有 ~230–250 ms 等效延迟；
`g(q_d + τ·q̇_d)` 用**一个标量**表达该超前，天然无过拟合、可解释。

**本工具**：在干净数据上按轴做 τ 的网格搜索（目标 = 最小化标签残差 SSE），LOSO 留出验证，
并与 τ=0（= 当前静态基线）对比留出 RMSE。

用法：
    python -m model.fit_velocity_lead --session <A> <B> <C> <D> \
        --base-config configs/static_feedforward_controller.json \
        --out configs/static_feedforward_lead_controller.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from control.controller_config import ControllerConfig


def _poly(c, x):
    return sum(cc * x ** k for k, cc in enumerate(c))


def load_session(csv_path: Path, threshold: float):
    """→ (q_d (n,2), q̇_d (n,2), u_total (n,2))，仅收敛段。

    速度在**全序列**上求（先排序）再筛收敛段——避免空洞伪峰（实验机修复的 bug）。
    """
    rows_all: dict[int, list] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        import csv as _csv
        for r in _csv.DictReader(f):
            sid = r.get("segment_id", "")
            if sid in ("", "-1", None):
                continue
            try:
                rows_all.setdefault(int(sid), []).append((
                    float(r["t_s"]),
                    float(r["target_front_back_deg"]), float(r["target_left_right_deg"]),
                    float(r["current_front_back_deg"]), float(r["current_left_right_deg"]),
                    float(r["servo_front_back_offset"]), float(r["servo_left_right_offset"]),
                ))
            except (KeyError, ValueError):
                continue
    Q, V, U = [], [], []
    for rows in rows_all.values():
        if len(rows) < 50:
            continue
        rows.sort(key=lambda x: x[0])
        ts = np.array([x[0] for x in rows])
        tf = np.array([x[1] for x in rows]); tl = np.array([x[2] for x in rows])
        cf = np.array([x[3] for x in rows]); cl = np.array([x[4] for x in rows])
        uf = np.array([x[5] for x in rows]); ul = np.array([x[6] for x in rows])
        dt = np.median(np.diff(ts))
        vf, vl = np.gradient(tf, dt), np.gradient(tl, dt)
        ok = (np.abs(cf - tf) < threshold) & (np.abs(cl - tl) < threshold)
        if ok.sum() < 50:
            continue
        Q.append(np.column_stack([tf[ok], tl[ok]]))
        V.append(np.column_stack([vf[ok], vl[ok]]))
        U.append(np.column_stack([uf[ok], ul[ok]]))
    if not Q:
        return None
    return np.concatenate(Q), np.concatenate(V), np.concatenate(U)


def sse_axis(cfg: ControllerConfig, ax: str, q, v, u, tau: float) -> float:
    """给定 τ，返回该轴标签残差 SSE：u_总 − g(q + τ·q̇)。"""
    ql = q + tau * v
    pred = np.array([_poly(cfg.gain_poly[ax], x) for x in ql])
    return float(np.sum((u - pred) ** 2))


def fit_linear(cfg, ax, q, v, u):
    """加性线性式 u_ff = g(q_d) + c·q̇_d 的最优 c（最小二乘，含偏置）。"""
    r = u - np.array([_poly(cfg.gain_poly[ax], x) for x in q])
    A = np.column_stack([np.ones_like(v), v])
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    rmse = float(np.sqrt(np.mean((r - A @ c) ** 2)))
    return float(c[1]), rmse


def fit_tau(cfg, ax, q, v, u, lo=-0.05, hi=0.45, n=201):
    """网格搜索 τ（1-D，粗搜 + 细搜）。"""
    taus = np.linspace(lo, hi, n)
    errs = [sse_axis(cfg, ax, q, v, u, t) for t in taus]
    t0 = taus[int(np.argmin(errs))]
    taus2 = np.linspace(t0 - 0.02, t0 + 0.02, 81)
    errs2 = [sse_axis(cfg, ax, q, v, u, t) for t in taus2]
    return float(taus2[int(np.argmin(errs2))]), float(np.min(errs2))


def main() -> int:
    ap = argparse.ArgumentParser(description="速度前馈 τ 拟合（u_ff = g(q_d + τ·q̇_d)）")
    ap.add_argument("--session", nargs="+", required=True, help="会话目录（≥3 供 LOSO）")
    ap.add_argument("--base-config", required=True, help="提供 g(q) 的静态前馈配置")
    ap.add_argument("--out", default=None, help="输出配置（含 velocity_lead）")
    ap.add_argument("--threshold", type=float, default=1.0, help="收敛阈值（度）")
    ap.add_argument("--train-all", action="store_true", help="用全部会话拟合（产物用）")
    args = ap.parse_args()

    cfg = ControllerConfig.load(args.base_config)
    if cfg.gain_poly is None:
        print("base-config 缺 gain_poly", file=sys.stderr)
        return 1
    parts = []
    for s in args.session:
        p = Path(s)
        d = p if p.is_dir() else p.parent
        r = load_session(d / "servo_data.csv", args.threshold)
        if r is None:
            print(f"跳过（无收敛段）: {d.name}", file=sys.stderr)
            continue
        parts.append(r)
        print(f"{d.name}: {len(r[0])} 收敛样本")
    if len(parts) < 2:
        print("有效会话不足 2", file=sys.stderr)
        return 1

    print("\n=== LOSO 留出验证：τ 由训练会话拟合，在留出会话评估 ===")
    print(f"{'轴':>3} {'τ_fit(s)':>9} {'RMSE(τ=0)':>11} {'RMSE(τ)':>10} {'改善':>8}  各折τ")
    taus_all = {"fb": [], "lr": []}
    for ax_i, ax in enumerate(["fb", "lr"]):
        r0s, r1s, taus = [], [], []
        for k in range(len(parts)):
            tr = [i for i in range(len(parts)) if i != k]
            q_tr = np.concatenate([parts[i][0][:, ax_i] for i in tr])
            v_tr = np.concatenate([parts[i][1][:, ax_i] for i in tr])
            u_tr = np.concatenate([parts[i][2][:, ax_i] for i in tr])
            q_te, v_te, u_te = parts[k][0][:, ax_i], parts[k][1][:, ax_i], parts[k][2][:, ax_i]
            tau, _ = fit_tau(cfg, ax, q_tr, v_tr, u_tr)
            taus.append(tau)
            r0s.append(np.sqrt(np.mean((u_te - np.array([_poly(cfg.gain_poly[ax], x) for x in q_te])) ** 2)))
            r1s.append(np.sqrt(np.mean((u_te - np.array([_poly(cfg.gain_poly[ax], x) for x in (q_te + tau * v_te)])) ** 2)))
        r0, r1 = float(np.mean(r0s)), float(np.mean(r1s))
        print(f"{ax:>3} {np.mean(taus):>9.4f} {r0:>11.2f} {r1:>10.2f} {(r1/r0-1)*100:>7.1f}%  "
              + " ".join(f"{t:.3f}" for t in taus))
        taus_all[ax] = taus

    # 最终（产物）：train-all 拟合
    q_all = [p[0] for p in parts]; v_all = [p[1] for p in parts]; u_all = [p[2] for p in parts]
    lead, detail = {}, {}
    for ax_i, ax in enumerate(["fb", "lr"]):
        q = np.concatenate([q[:, ax_i] for q in q_all])
        v = np.concatenate([v[:, ax_i] for v in v_all])
        u = np.concatenate([u[:, ax_i] for u in u_all])
        tau, sse = fit_tau(cfg, ax, q, v, u)
        lead[ax] = tau
        rmse0 = np.sqrt(np.mean((u - np.array([_poly(cfg.gain_poly[ax], x) for x in q])) ** 2))
        rmse1 = np.sqrt(np.mean((u - np.array([_poly(cfg.gain_poly[ax], x) for x in (q + tau * v)])) ** 2))
        detail[ax] = dict(tau=tau, rmse_tau0=float(rmse0), rmse_tau=float(rmse1))
        print(f"\n全量拟合 [{ax}] τ = {tau:.4f} s   残差 RMSE {rmse0:.2f} → {rmse1:.2f} "
              f"({(rmse1/rmse0-1)*100:+.1f}%)")

    # 加性线性式（D0 实测最优形式）：拟合 c 并与 τ 式对比
    print("\n=== 形式对照（离线 RMSE，offset）：τ=0 vs 加性 c·q̇ vs 相位超前 τ ===")
    gain = {}
    for ax_i, ax in enumerate(["fb", "lr"]):
        q = np.concatenate([p[0][:, ax_i] for p in parts])
        v = np.concatenate([p[1][:, ax_i] for p in parts])
        u = np.concatenate([p[2][:, ax_i] for p in parts])
        c_lin, rmse_lin = fit_linear(cfg, ax, q, v, u)
        tau, _ = fit_tau(cfg, ax, q, v, u)
        rmse0 = float(np.sqrt(np.mean((u - np.array([_poly(cfg.gain_poly[ax], x) for x in q])) ** 2)))
        rmse_tau = float(np.sqrt(np.mean((u - np.array([_poly(cfg.gain_poly[ax], x) for x in (q + tau * v)])) ** 2)))
        gain[ax] = c_lin
        print(f"  [{ax}] τ=0: {rmse0:6.2f} | 加性 c={c_lin:+.3f} → {rmse_lin:6.2f} | "
              f"τ={tau:.4f}s → {rmse_tau:6.2f}  → 优者：{'加性线性' if rmse_lin < rmse_tau else '相位超前 τ'}")
    print(f"\n→ velocity_gain = {json.dumps(gain)}")
    print(f"→ velocity_lead = {json.dumps(lead)}")
    if args.out:
        data = json.loads(Path(args.base_config).read_text(encoding="utf-8-sig"))
        data["velocity_lead"] = lead
        data["velocity_gain"] = gain
        data["_note"] = (f"速度前馈（两种形式并存，A/B 二选一）："
                         f"① 加性线性 velocity_gain c_fb={gain['fb']:+.3f} c_lr={gain['lr']:+.3f} "
                         f"offset/(°/s)（D0 干净数据实测更优）；"
                         f"② 相位超前 velocity_lead τ_fb={lead['fb']:.4f}s τ_lr={lead['lr']:.4f}s"
                         f"（等价系数 g′(q)·τ，带 q 依赖，离线 RMSE 差约 48%）。"
                         f"由 {len(parts)} 个干净 D0 会话拟合（slew 20 基线）；"
                         f"两式的系数置零均逐位退回 static_feedforward")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写: {out}")
    print("\n⚠️ 离线指标不作判据——merge 由多轨迹实机 A/B 决定（≥3 轨迹，含 Eight 类 1:2 相位）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
