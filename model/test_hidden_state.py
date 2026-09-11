"""T1：可证伪实验——张力特征是否解释了"隐状态"（§8.4 架构选型关键判据）。

**假设 H1**：无记忆模型（D1）学不好的"记忆"，主要来自腱的**松弛/张紧隐状态**，
而张力传感器可观测它。若成立 → 无记忆模型 + 张力特征可能就够（不必上 D2 序列模型）；
若不成立 → 张力不是那个隐状态，需要 D2（历史窗）。

**判据**：在【状态 (q, q̇) 已被充分建模】之后，张力特征能否解释 Δu* 的**额外**方差？
    Δu*(t) = u_总(t) − g_static(q_d(t))      （收敛段：u_总 ≈ 目标状态所需 u）

**三组控制（缺一不可，否则结论不可信）**：
    1. **打乱张力**（行置换）→ 额外 R² 必须 ≈0（排除"多特征必然涨 R²"）
    2. **留出会话交叉验证**（LOSO）——不是样本内（级1.6 教训：样本内必涨）
    3. **多会话一致**——单条轨迹结论会骗人（D1 教训）

用法：
    python -m model.test_hidden_state --session <A> <B> <C> <D> \
        --base-config configs/static_feedforward_controller.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from control.controller_config import ControllerConfig

FORCE_COLS = ["ch1", "ch2", "ch3", "ch4"]


# ---------------------------------------------------------------------------
# 数据加载：servo(状态/指令) + force(张力) 按时间戳对齐
# ---------------------------------------------------------------------------

def _read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _base_u(cfg: ControllerConfig, q_fb: float, q_lr: float) -> tuple[float, float]:
    """静态基座 g_static（不含 dynamic_nn / 迟滞）。"""
    def poly(c, x):
        return sum(cc * x ** k for k, cc in enumerate(c))
    if cfg.gain_poly is not None:
        return poly(cfg.gain_poly["fb"], q_fb), poly(cfg.gain_poly["lr"], q_lr)
    if cfg.direction_gains is not None:
        g = cfg.direction_gains
        return (q_fb / (g["fb"]["pos"] if q_fb >= 0 else g["fb"]["neg"]),
                q_lr / (g["lr"]["pos"] if q_lr >= 0 else g["lr"]["neg"]))
    return 0.0, 0.0


def load_dyn(session_dir: Path, cfg: ControllerConfig, threshold: float):
    """前向模型样本：(q,u,q̇,q̇_next,T) —— 用于**非循环**的前向检验。

    返回 (X_base (n,7)=[q_fb,q_lr,v_fb,v_lr,u_fb,u_lr,Δq_next 的预测目标...], T, y)。
    这里返回 (q (n,2), v (n,2), u (n,2), y=Δq_next (n,2), T (n,4))。
    """
    try:
        srows = [r for r in _read_csv(session_dir / "servo_data.csv")
                 if r.get("segment_id", "") not in ("", "-1", None)]
        frows = _read_csv(session_dir / "force_data.csv")
    except FileNotFoundError:
        return None
    if not srows or not frows:
        return None
    fts = np.array([int(r["pc_receive_unix_time_ms"]) for r in frows])
    F = np.array([[float(r[c]) for c in FORCE_COLS] for r in frows])
    o = np.argsort(fts)
    fts, F = fts[o], F[o]

    per_seg: dict[int, list] = {}
    for r in srows:
        per_seg.setdefault(int(r["segment_id"]), []).append(r)

    Q, V, U, Y, TT = [], [], [], [], []
    for rows in per_seg.values():
        rows.sort(key=lambda r: float(r["t_s"]))
        ts = np.array([int(r["pc_receive_unix_time_ms"]) for r in rows])
        tsec = np.array([float(r["t_s"]) for r in rows])
        qf = np.array([float(r["current_front_back_deg"]) for r in rows])
        ql = np.array([float(r["current_left_right_deg"]) for r in rows])
        uf = np.array([float(r["servo_front_back_offset"]) for r in rows])
        ul = np.array([float(r["servo_left_right_offset"]) for r in rows])
        if len(rows) < 100:
            continue
        dt = np.median(np.diff(tsec))
        vf, vl = np.gradient(qf, dt), np.gradient(ql, dt)
        idx = np.clip(np.searchsorted(fts, ts), 0, len(fts) - 1)
        Fi = F[idx]
        n = len(rows) - 1
        Q.append(np.column_stack([qf[:n], ql[:n]]))
        V.append(np.column_stack([vf[:n], vl[:n]]))
        U.append(np.column_stack([uf[:n], ul[:n]]))
        Y.append(np.column_stack([qf[1:] - qf[:n], ql[1:] - ql[:n]]))
        TT.append(Fi[:n])
    if not Q:
        return None
    return (np.concatenate(Q), np.concatenate(V), np.concatenate(U),
            np.concatenate(Y), np.concatenate(TT))


def load_aligned(session_dir: Path, cfg: ControllerConfig, threshold: float):
    """→ (state (n,4)=[q_fb,q_lr,q̇_fb,q̇_lr], Δu* (n,2), tens (n,4))。"""
    try:
        srows = [r for r in _read_csv(session_dir / "servo_data.csv")
                 if r.get("segment_id", "") not in ("", "-1", None)]
        frows = _read_csv(session_dir / "force_data.csv")
    except FileNotFoundError:
        return None
    if not srows or not frows:
        return None
    fts = np.array([int(r["pc_receive_unix_time_ms"]) for r in frows])
    F = np.array([[float(r[c]) for c in FORCE_COLS] for r in frows])
    order = np.argsort(fts)
    fts, F = fts[order], F[order]

    per_seg: dict[int, list] = {}
    for r in srows:
        per_seg.setdefault(int(r["segment_id"]), []).append(r)

    S, DU, T = [], [], []
    for rows in per_seg.values():
        rows.sort(key=lambda r: float(r["t_s"]))
        ts = np.array([int(r["pc_receive_unix_time_ms"]) for r in rows])
        tsec = np.array([float(r["t_s"]) for r in rows])
        tf = np.array([float(r["target_front_back_deg"]) for r in rows])
        tl = np.array([float(r["target_left_right_deg"]) for r in rows])
        cf = np.array([float(r["current_front_back_deg"]) for r in rows])
        cl = np.array([float(r["current_left_right_deg"]) for r in rows])
        uf = np.array([float(r["servo_front_back_offset"]) for r in rows])
        ul = np.array([float(r["servo_left_right_offset"]) for r in rows])
        ok = (np.abs(cf - tf) < threshold) & (np.abs(cl - tl) < threshold)
        if ok.sum() < 50:
            continue
        dt = np.median(np.diff(tsec))
        vf, vl = np.gradient(tf, dt), np.gradient(tl, dt)   # 解析参考速度
        idx = np.clip(np.searchsorted(fts, ts), 0, len(fts) - 1)
        j = np.where(ok)[0]
        if len(j) < 50:
            continue
        du = np.array([[uf[i] - _base_u(cfg, tf[i], tl[i])[0],
                        ul[i] - _base_u(cfg, tf[i], tl[i])[1]] for i in j])
        S.append(np.column_stack([tf[j], tl[j], vf[j], vl[j]]))
        DU.append(du)
        T.append(F[idx][j])
    if not S:
        return None
    return np.concatenate(S), np.concatenate(DU), np.concatenate(T)


# ---------------------------------------------------------------------------
# 特征
# ---------------------------------------------------------------------------

_STATE_NAMES = ["q_fb", "q_lr", "v_fb", "v_lr"]


def state_features(S: np.ndarray, degree: int):
    cols, names = [np.ones(len(S))], ["1"]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(4), d):
            c = np.ones(len(S))
            for k in combo:
                c = c * S[:, k]
            cols.append(c)
            names.append("*".join(_STATE_NAMES[k] for k in combo))
    cols += [np.sign(S[:, 2]), np.sign(S[:, 3])]
    names += ["sign(v_fb)", "sign(v_lr)"]
    return np.column_stack(cols), names


def tension_features(T: np.ndarray):
    c1, c2, c3, c4 = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
    f = np.column_stack([np.minimum(c1, c3), np.minimum(c2, c4),
                         c1 - c3, c2 - c4,
                         0.5 * (c1 + c3), 0.5 * (c2 + c4)])
    return f, ["min_fb", "min_lr", "diff_fb", "diff_lr", "comm_fb", "comm_lr"]


def _stdz(tr, te):
    m, s = tr.mean(0), tr.std(0) + 1e-9
    return (tr - m) / s, (te - m) / s


def _ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def _r2(y, pred):
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 0.0


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------

def run(sessions, cfg, threshold, degree, lam, seed=0):
    rng = np.random.default_rng(seed)
    parts = [load_aligned(p, cfg, threshold) for p in sessions]
    keep = [(sessions[i], p) for i, p in enumerate(parts) if p is not None]
    if len(keep) < 3:
        print(f"有效会话不足 3（只拿到 {len(keep)} 个；需 ≥3 才能留出+多轨迹一致）", file=sys.stderr)
        return 1
    sessions = [k[0] for k in keep]
    parts = [k[1] for k in keep]
    S = np.concatenate([p[0] for p in parts])
    DU = np.concatenate([p[1] for p in parts])
    T = np.concatenate([p[2] for p in parts])
    sid = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])
    n = len(parts)
    print(f"\n会话 {n} 个，收敛样本 {len(S)}")
    for i, p in enumerate(parts):
        print(f"  #{i} {sessions[i].name}: {len(p[0])} 样本")

    Xs, _ = state_features(S, degree)
    Xt, _ = tension_features(T)
    print(f"状态特征 {Xs.shape[1]}（{degree} 阶多项式+sign） 张力特征 {Xt.shape[1]}")

    summary = {}
    for axis, axname in [(0, "fb"), (1, "lr")]:
        y = DU[:, axis]
        rec = {k: [] for k in ("base", "extra", "shuf", "self")}
        for k in range(n):
            tr, te = sid != k, sid == k
            if te.sum() < 100:
                continue
            Xtr, Xte = _stdz(Xs[tr], Xs[te])
            w = _ridge(Xtr, y[tr], lam)
            pred = Xte @ w
            rec["base"].append(_r2(y[te], pred))
            eps_tr = y[tr] - Xtr @ w                     # 训练残差
            Xttr, Xtte = _stdz(Xt[tr], Xt[te])
            rec["extra"].append(_r2(y[te], pred + Xtte @ _ridge(Xttr, eps_tr, lam)))
            # 控制①：**只打乱训练目标**（特征不动）→ 打断特征-目标关系，额外应≈0
            p1 = rng.permutation(len(eps_tr))
            rec["shuf"].append(_r2(y[te], pred + Xtte @ _ridge(Xttr, eps_tr[p1], lam)))
            # 控制②：**同维随机特征** → 检验"任意 k 个特征都会涨 R²"的假象
            Rtr, Rte = _stdz(rng.normal(size=(len(eps_tr), Xt.shape[1])),
                             rng.normal(size=(int(te.sum()), Xt.shape[1])))
            rec["self"].append(_r2(y[te], pred + Rte @ _ridge(Rtr, eps_tr, lam)))
        base = np.mean(rec["base"]); extra = np.mean(rec["extra"])
        c1 = np.mean(rec["shuf"]); c2 = np.mean(rec["self"])
        print(f"\n=== 轴 {axname}（留出会话平均，{len(rec['base'])} 折）===")
        print(f"  ① 状态模型 R²(留出)          = {base:+.4f}")
        print(f"  ② + 张力 → R²(留出)          = {extra:+.4f}   额外 = {extra-base:+.4f}")
        print(f"  控制①打乱张力 → 额外          = {c1-base:+.4f}")
        print(f"  控制②同维随机特征 → 额外      = {c2-base:+.4f}")
        gain, k1, k2 = extra - base, c1 - base, c2 - base
        ok = gain > 0.02 and gain > 3 * abs(k1) and gain > 3 * abs(k2)
        print(f"  判定: {'张力带来额外信息 ✅' if ok else '未超过控制组 → 张力未证明解释隐状态 ❌'}")
        summary[axname] = gain

    # ---- 决定性检验（非循环）：前向模型 —— 张力能否帮助预测"下一步运动"？ ----
    # 循环性说明：Δu* 本身驱动张力 ⇒ 用并发张力预测 Δu* 是循环论证（上面细格检验即栽在此）。
    # 正确问法：给定 (q, q̇, u)，**张力是否提供了额外的被控对象状态信息**（摩擦/松弛隐状态），
    # 从而能更好地预测下一拍位移 Δq_next？张力是【过去指令的后果】，预测的是【未来】→ 非循环。
    print("\n=== 决定性检验（非循环）：前向模型 q(t+1) 预测，张力是否带来增量 ===")
    dyn = [load_dyn(p, cfg, threshold) for p in sessions]
    dyn = [d for d in dyn if d is not None]
    if len(dyn) >= 3:
        Q = np.concatenate([d[0] for d in dyn]); V = np.concatenate([d[1] for d in dyn])
        U = np.concatenate([d[2] for d in dyn]); Y = np.concatenate([d[3] for d in dyn])
        TT = np.concatenate([d[4] for d in dyn])
        dsid = np.concatenate([np.full(len(d[0]), i) for i, d in enumerate(dyn)])
        base_in = np.column_stack([Q, V, U])
        Xb, _ = state_features(base_in, degree)
        Xt2, _ = tension_features(TT)
        print(f"  前向样本 {len(Q)}  基线特征 {Xb.shape[1]}（q,v,u 多项式）  张力 {Xt2.shape[1]}")
        for axis, axname in [(0, "fb"), (1, "lr")]:
            y = Y[:, axis]
            rec = {k: [] for k in ("base", "extra", "shuf", "rand")}
            for k in range(len(dyn)):
                tr, te = dsid != k, dsid == k
                if te.sum() < 100:
                    continue
                Xtr, Xte = _stdz(Xb[tr], Xb[te])
                w = _ridge(Xtr, y[tr], lam)
                pred = Xte @ w
                rec["base"].append(_r2(y[te], pred))
                eps = y[tr] - Xtr @ w
                Ttr, Tte = _stdz(Xt2[tr], Xt2[te])
                rec["extra"].append(_r2(y[te], pred + Tte @ _ridge(Ttr, eps, lam)))
                p = rng.permutation(len(eps))
                rec["shuf"].append(_r2(y[te], pred + Tte @ _ridge(Ttr, eps[p], lam)))
                Rt, Re = _stdz(rng.normal(size=(len(eps), Xt2.shape[1])),
                               rng.normal(size=(int(te.sum()), Xt2.shape[1])))
                rec["rand"].append(_r2(y[te], pred + Re @ _ridge(Rt, eps, lam)))
            b, e = np.mean(rec["base"]), np.mean(rec["extra"])
            c1, c2 = np.mean(rec["shuf"]), np.mean(rec["rand"])
            g = e - b
            ok = g > 0.005 and g > 3 * abs(c1 - b) and g > 3 * abs(c2 - b)
            print(f"  [{axname}] 基线 R²={b:+.4f}  +张力 R²={e:+.4f}  增量={g:+.4f}"
                  f"  控制(打乱)={c1-b:+.4f} 控制(随机)={c2-b:+.4f}")
            print(f"        判定: {'张力携带被控对象隐状态信息 ✅' if ok else '张力未带来前向预测增量 ❌'}")
    else:
        print("  （有效会话不足，跳过）")

    # ---- 参考：细格内相关（**注意：此项有循环性，仅作对比**）----
    # 把 4 维状态 (q_fb,q_lr,v_fb,v_lr) 切细格；格内状态近乎不变，
    # 此时 Δu* 若仍与张力相关 → 张力携带"状态之外"的隐状态信息（无容量混淆）。
    # 控制：格内打乱张力 → 同一统计量应归零。
    print("\n=== 细格（4 维状态）内：Δu* ~ 张力差模 相关（不依赖状态模型形式）===")
    nb = 4
    edges = [np.quantile(S[:, k], np.linspace(0, 1, nb + 1)[1:-1]) for k in range(4)]
    cell = np.column_stack([np.digitize(S[:, k], edges[k]) for k in range(4)])
    print(f"  分箱 {nb}^4={nb**4} 格，样本 {len(S)}")
    for axis, axname in [(0, "fb"), (1, "lr")]:
        feat = Xt[:, 2 + axis]                      # diff_fb / diff_lr
        real, shuf, ns = [], [], []
        for c in np.unique(cell, axis=0):
            m = np.all(cell == c, axis=1)
            if m.sum() < 30:
                continue
            y, x = DU[m, axis], feat[m]
            if y.std() < 1e-9 or x.std() < 1e-9:
                continue
            real.append(np.corrcoef(y, x)[0, 1])
            shuf.append(np.corrcoef(y, rng.permutation(x))[0, 1])
            ns.append(int(m.sum()))
        real, shuf, ns = np.array(real), np.array(shuf), np.array(ns)
        if not len(real):
            continue
        w = np.sqrt(ns)
        mr, ms = np.average(real, weights=w), np.average(shuf, weights=w)
        ok = abs(mr) > 5 * abs(ms) and abs(mr) > 0.1
        print(f"  [{axname}] 有效格数={len(real)}  平均格内样本={ns.mean():.0f}")
        print(f"        真实 平均 corr(Δu*, diff_{axname}) = {mr:+.4f}（加权）"
              f"  |corr|>0.3 占比={100*np.mean(np.abs(real)>0.3):.0f}%")
        print(f"        控制 打乱后平均 corr             = {ms:+.4f}（应≈0）")
        print(f"        判定: {'格内张力确实携带状态外信息 ✅' if ok else '格内无明显额外信息 ❌'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="T1：张力特征是否解释隐状态")
    ap.add_argument("--session", nargs="+", required=True, help="会话目录（≥3，需留出）")
    ap.add_argument("--base-config", required=True, help="静态前馈配置（提供 g_static）")
    ap.add_argument("--threshold", type=float, default=0.5, help="收敛阈值（度）")
    ap.add_argument("--degree", type=int, default=2, help="状态多项式阶数")
    ap.add_argument("--lam", type=float, default=1e-2, help="ridge 正则")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = ControllerConfig.load(args.base_config)
    paths = []
    for s in args.session:
        p = Path(s)
        paths.append(p if p.is_dir() else p.parent)
    return run(paths, cfg, args.threshold, args.degree, args.lam, args.seed)


if __name__ == "__main__":
    sys.exit(main())
