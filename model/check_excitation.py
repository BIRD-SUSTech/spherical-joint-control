"""检查轨迹对 (q, q̇, q̈) 的激励质量：q–q̈ 可辨识性（VIF）+ 加速度覆盖。

动机（T1 结论）：lr 换向过冲指向**未补偿的减速/惯性需求 q̈**，但历次回归都测不到 q̈，因为
    ① 单频 sine 上 q̈ = −ω²q → q 与 q̈ 共线（corr≈−1），不可辨识；
    ② 2D 网格 |q̈| 极小（实测 p99 仅 1.4°/s²）→ 根本没激励。
所以 D0 采集后必须**先验证激励质量**再谈建模。

VIF（方差膨胀因子）越大 = 共线越严重、越不可辨识：
    VIF_j = 1 / (1 − R²_j)，R²_j 来自把第 j 个回归量对其余回归量做最小二乘。
    VIF=1 正交（最好）；VIF>5 通常视为共线严重。

用法：
    python -m model.check_excitation --session <会话1> [<会话2> ...]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_refs(session: Path):
    """读参考轨迹（全序列，均匀 dt）→ {seg: (t, qf, ql, vf, vl, af, al)}。"""
    segs: dict[int, list] = {}
    with open(session / "servo_data.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r.get("segment_id", "")
            if sid in ("", "-1", None):
                continue
            segs.setdefault(int(sid), []).append(
                (float(r["t_s"]), float(r["target_front_back_deg"]), float(r["target_left_right_deg"])))
    out = {}
    for g, rows in segs.items():
        if len(rows) < 10:
            continue
        rows.sort()
        d = np.array(rows)
        t, qf, ql = d[:, 0], d[:, 1], d[:, 2]
        dt = np.median(np.diff(t))
        vf, vl = np.gradient(qf, dt), np.gradient(ql, dt)
        af, al = np.gradient(vf, dt), np.gradient(vl, dt)
        out[g] = (t, qf, ql, vf, vl, af, al)
    return out


def vif(X: np.ndarray) -> np.ndarray:
    """X (n,k) → 各列 VIF (k,)。"""
    k = X.shape[1]
    out = np.full(k, np.inf)
    for j in range(k):
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        resid = X[:, j] - A @ coef
        ss_tot = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
        ss_res = float((resid ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[j] = 1.0 / max(1e-12, 1.0 - min(r2, 0.999999))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="轨迹激励质量检查（VIF / |q̈| 覆盖）")
    ap.add_argument("--session", nargs="+", required=True, help="会话目录（可多个，合并评估）")
    args = ap.parse_args()

    print("%-22s %-9s %-24s %-26s %s" % ("会话", "段", "VIF (q,q̇,q̈)", "corr(q,q̈) fb/lr", "|q̈| p99 fb/lr  |q̇| 峰值 fb/lr"))
    agg = {"fb": [], "lr": []}
    for s in args.session:
        p = Path(s)
        p = p if p.is_dir() else p.parent
        try:
            segs = load_refs(p)
        except FileNotFoundError:
            print("%-22s 跳过（无 servo_data.csv）" % p.name)
            continue
        for g, (t, qf, ql, vf, vl, af, al) in sorted(segs.items()):
            row = []
            for q, v, a in ((qf, vf, af), (ql, vl, al)):
                X = np.column_stack([q, v, a])
                row.append(vif(X))
                agg["fb" if q is qf else "lr"].append((q, v, a))
            vf_ = row[0]
            vl_ = row[1]
            c_f = float(np.corrcoef(qf, af)[0, 1])
            c_l = float(np.corrcoef(ql, al)[0, 1])
            print("%-22s seg%-6d %5.2f/%5.2f/%5.2f      %+.3f / %+.3f        %6.1f /%6.1f   %5.1f /%5.1f" % (
                p.name, g, vf_[0], vf_[1], vf_[2], c_f, c_l,
                np.percentile(np.abs(af), 99), np.percentile(np.abs(al), 99),
                np.abs(vf).max(), np.abs(vl).max()))

    print()
    for name, lst in agg.items():
        if not lst:
            continue
        X = np.vstack([np.column_stack(x) for x in lst])
        if len(X) < 50:
            continue
        v = vif(X)
        print("合并 %s: VIF(q)=%.2f VIF(q̇)=%.2f VIF(q̈)=%.2f  ← 最大 %.2f %s" % (
            name, v[0], v[1], v[2], v.max(),
            "✅ 可辨识" if v.max() < 2 else ("△ 偏共线" if v.max() < 5 else "❌ 共线严重")))
    print("\n判据：最大 VIF < 2（q̈ 可辨识）+ |q̈| p99 足够大（q̈ 被激励）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
