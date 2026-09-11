"""动态项辨识：Δu* 对 (q, q̇, q̈, sign(q̇)) 的回归 —— 判定 q̈ 项是否值得进模型。

**背景**（T1 追加分析）：lr 换向过冲 +3~5°（幅值 20%），根因指向**未补偿的减速/惯性需求 q̈**；
但历次回归测不到 q̈，因为旧数据 ①单频 sine 上 q̈=−ω²q（退化，VIF→∞）②2D 网格 |q̈|p99 仅 1.4°/s²。
D0 第二轮采集专门解决此问题（speed-ladder/chirp/random-fourier，合并 VIF 1.28 < 2）。

**本工具回答**：加入 q̈ 项后，**留出会话**的 R² 是否显著提升？是否超过两个控制组？
    ① 打乱 q̈（只打乱训练目标）→ 应≈0
    ② 同维随机特征 → 应≈0
    ③ 阶数稳健性：低阶/高阶状态模型下结论是否一致（T1 教训：容量混淆）

**关键实现纪律**（继承实验机修复的 bug）：
    参考速度/加速度必须在【全序列】上求（先排序求梯度）**再**筛收敛段——
    否则序列空洞会造出上千 °/s 的伪峰。

用法：
    python -m model.fit_dynamic_terms --session <A> <B> <C> <D> \
        --base-config configs/static_feedforward_controller.json [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from control.controller_config import ControllerConfig


def _poly(c, x):
    return sum(cc * x ** k for k, cc in enumerate(c))


def base_static_u(cfg: ControllerConfig, q_fb: float, q_lr: float):
    if cfg.gain_poly is not None:
        return _poly(cfg.gain_poly["fb"], q_fb), _poly(cfg.gain_poly["lr"], q_lr)
    if cfg.direction_gains is not None:
        g = cfg.direction_gains
        return (q_fb / (g["fb"]["pos"] if q_fb >= 0 else g["fb"]["neg"]),
                q_lr / (g["lr"]["pos"] if q_lr >= 0 else g["lr"]["neg"]))
    return 0.0, 0.0


def load_session(csv_path: Path, cfg: ControllerConfig, threshold: float):
    """→ dict(q=[q_fb,q_lr], v=[...], a=[...], du=[...])，只在【收敛段】返回样本。

    速度/加速度在**全序列**上求（先排序），再筛收敛段（避免空洞伪峰）。
    """
    rows_all: dict[int, list] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
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

    Q, V, A, DU = [], [], [], []
    for rows in rows_all.values():
        if len(rows) < 50:
            continue
        rows.sort(key=lambda x: x[0])
        ts = np.array([x[0] for x in rows])
        tf = np.array([x[1] for x in rows]); tl = np.array([x[2] for x in rows])
        cf = np.array([x[3] for x in rows]); cl = np.array([x[4] for x in rows])
        uf = np.array([x[5] for x in rows]); ul = np.array([x[6] for x in rows])
        dt = np.median(np.diff(ts))
        # 全序列求导（目标解析平滑 → 干净）
        vf, vl = np.gradient(tf, dt), np.gradient(tl, dt)
        af, al = np.gradient(vf, dt), np.gradient(vl, dt)
        ok = (np.abs(cf - tf) < threshold) & (np.abs(cl - tl) < threshold)
        if ok.sum() < 50:
            continue
        for i in np.where(ok)[0]:
            bf, bl = base_static_u(cfg, tf[i], tl[i])
            Q.append([tf[i], tl[i]]); V.append([vf[i], vl[i]]); A.append([af[i], al[i]])
            DU.append([uf[i] - bf, ul[i] - bl])
    if not Q:
        return None
    return {k: np.asarray(v) for k, v in
            dict(q=Q, v=V, a=A, du=DU).items()}


def features(d, with_a: bool, degree: int):
    q, v = d["q"], d["v"]
    cols = [np.ones(len(q)), q[:, 0], q[:, 1], v[:, 0], v[:, 1],
            np.sign(v[:, 0]), np.sign(v[:, 1])]
    for k in (2, 3):
        if k <= degree:
            cols += [q[:, 0] ** k, q[:, 1] ** k]
    if degree >= 2:
        cols += [q[:, 0] * q[:, 1], v[:, 0] * q[:, 0], v[:, 1] * q[:, 1]]
    if degree >= 3:
        cols += [q[:, 0] ** 2 * v[:, 0], q[:, 1] ** 2 * v[:, 1], q[:, 0] * q[:, 1] * v[:, 0]]
    if with_a:
        cols += [d["a"][:, 0], d["a"][:, 1]]
    return np.column_stack(cols)


def _stdz(tr, te):
    m, s = tr.mean(0), tr.std(0) + 1e-9
    return (tr - m) / s, (te - m) / s


def _ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def _r2(y, p):
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else 0.0


def run(paths, cfg, threshold, degree, lam, seed=0):
    rng = np.random.default_rng(seed)
    parts = [load_session(p / "servo_data.csv" if p.is_dir() else p, cfg, threshold) for p in paths]
    keep = [(paths[i], p) for i, p in enumerate(parts) if p]
    if len(keep) < 3:
        print(f"有效会话不足 3（得 {len(keep)}）", file=sys.stderr)
        return 1, None
    paths = [k[0] for k in keep]; parts = [k[1] for k in keep]
    n = len(parts)
    print(f"\n会话 {n} 个，收敛样本 {sum(len(p['q']) for p in parts)}")
    for i, p in enumerate(parts):
        print(f"  #{i} {paths[i].name}: {len(p['q'])}")

    # 四种特征矩阵（**控制组 = 基线 + 无用特征**，直接回答"增益是否来自 q̈"）：
    #   base   : 无 q̈
    #   full   : 基线 + 真 q̈
    #   shuf   : 基线 + 被打乱的 q̈（同维、无信息）
    #   rand   : 基线 + 同维随机列
    rng2 = np.random.default_rng(seed + 1)
    variants = ["base", "full", "shuf", "rand"]
    out = {}
    for axis, axname in [(0, "fb"), (1, "lr")]:
        Xs, ys = {v: [] for v in variants}, []
        for p in parts:
            Xb = features(p, False, degree)
            a = p["a"]
            Xs["base"].append(Xb)
            Xs["full"].append(np.column_stack([Xb, a[:, 0], a[:, 1]]))
            Xs["shuf"].append(np.column_stack([Xb, rng2.permutation(a[:, 0]),
                                               rng2.permutation(a[:, 1])]))
            Xs["rand"].append(np.column_stack([Xb, rng2.normal(size=len(a)),
                                               rng2.normal(size=len(a))]))
            ys.append(p["du"][:, axis])

        r2 = {v: [] for v in variants}
        folds = 0
        for k in range(n):
            tr = [i for i in range(n) if i != k]
            yte = ys[k]
            if len(yte) < 50:
                continue
            folds += 1
            ytr = np.concatenate([ys[i] for i in tr])
            for v in variants:
                Xtr = np.concatenate([Xs[v][i] for i in tr])
                Xtr_s, Xte_s = _stdz(Xtr, Xs[v][k])
                r2[v].append(_r2(yte, Xte_s @ _ridge(Xtr_s, ytr, lam)))
        m = {v: (float(np.mean(r2[v])) if r2[v] else float("nan")) for v in variants}
        gain, g_shuf, g_rand = m["full"] - m["base"], m["shuf"] - m["base"], m["rand"] - m["base"]
        ok = (gain > 0.005 and gain > 3 * abs(g_shuf) and gain > 3 * abs(g_rand))
        print(f"\n=== 轴 {axname}（degree={degree}, LOSO {folds} 折）===")
        print(f"  基线(无 q̈)      R²(留出) = {m['base']:+.4f}")
        print(f"  +真 q̈           R²(留出) = {m['full']:+.4f}   增量 = {gain:+.4f}")
        print(f"  控制① 基线+打乱q̈ R²(留出) = {m['shuf']:+.4f}   增量 = {g_shuf:+.4f}")
        print(f"  控制② 基线+随机列 R²(留出) = {m['rand']:+.4f}   增量 = {g_rand:+.4f}")
        print(f"  判定: {'q̈ 增益显著超过控制组 ✅' if ok else 'q̈ 增益未超过控制组 ❌'}")
        out[axname] = dict(base=m["base"], full=m["full"], gain=gain,
                           gain_shuf=g_shuf, gain_rand=g_rand, ok=bool(ok))
    return 0, out


def main() -> int:
    ap = argparse.ArgumentParser(description="动态项辨识：q̈ 项是否值得进模型")
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = ControllerConfig.load(args.base_config)
    paths = [Path(s) for s in args.session]
    rc, out = run(paths, cfg, args.threshold, args.degree, args.lam, args.seed)
    if args.json and out:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写: {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
