"""A/B 相位滞后报告：把跟踪误差翻译成"等效时间延迟"，并画"对齐前后"对比图。

动机：闭环跟踪误差常常 90%+ 是【切向】分量（纯相位滞后）。在轨迹俯视图/动图上，
相位滞后表现为"同一个形状被转了一下" —— 肉眼看不出差异，但指标差好几倍。
本工具给出人眼可判读的两件事：
    1. 等效时间延迟 τ（秒/毫秒）—— 用互相关在目标与实际之间求最优时移；
    2. 对齐前后对比图 —— 把实际轨迹按 τ 平移后应与目标重合，"差异 = 纯延迟"一眼可见。

用法：
    python -m valuation.phase_report <会话目录> [--segment-raw 0 1] [--out 输出.png]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(session: Path):
    segs = {}
    with open(session / "servo_data.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r.get("segment_id", "")
            if sid in ("", "-1", None):
                continue
            segs.setdefault(int(sid), []).append(
                (float(r["t_s"]), float(r["target_front_back_deg"]), float(r["target_left_right_deg"]),
                 float(r["current_front_back_deg"]), float(r["current_left_right_deg"])))
    return {g: np.array(sorted(v)) for g, v in segs.items()}


def best_delay(t, qd, q, max_lag_s=1.0):
    """互相关求等效时间延迟（s）：使 q(t+τ) 与 q_d(t) 最匹配的 τ。"""
    dt = np.median(np.diff(t))
    n = int(max_lag_s / dt)
    a = qd - qd.mean()
    b = q - q.mean()
    best_tau, best_c = 0.0, -2.0
    for k in range(-n, n + 1):
        if k >= 0:
            x, y = a[:len(a) - k] if k else a, b[k:]
        else:
            x, y = a[-k:], b[:len(b) + k]
        if len(x) < 50:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best_c:
            best_c, best_tau = c, k * dt
    return best_tau, best_c


def shift_by(t, q, tau):
    return t + tau, q


def reversal_overshoot(t, qd, q, win_s=0.25, min_gap=0.2):
    """换向点（q̇ 变号）附近 ±win_s 内的峰值过冲。

    err = q_d − q。在 +侧换向（由正转负）时 err<0 表示球冲过目标（过冲）。
    返回 (n_rev, 平均带符号峰值, 平均 |峰值|, 最大 |峰值|)。
    """
    dt = np.median(np.diff(t))
    v = np.gradient(qd, dt)
    s = np.sign(v)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return 0, 0.0, 0.0, 0.0
    # 去重：换向点间隔 < min_gap 的合并
    keep = [idx[0]]
    for i in idx[1:]:
        if (t[i] - t[keep[-1]]) > min_gap:
            keep.append(i)
    n = int(win_s / dt)
    peaks = []
    for i in keep:
        w = slice(max(0, i - n), min(len(t), i + n))
        e = qd[w] - q[w]
        peaks.append(e[np.argmax(np.abs(e))])
    p = np.array(peaks)
    return len(peaks), float(p.mean()), float(np.abs(p).mean()), float(np.abs(p).max())


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B 相位滞后报告（等效延迟 + 换向过冲 + 对齐对比图）")
    ap.add_argument("session")
    ap.add_argument("--t-min", type=float, default=5.0, help="稳态起点 s")
    ap.add_argument("--out", default=None, help="输出 PNG")
    ap.add_argument("--json", action="store_true", help="同时打印 JSON")
    ap.add_argument("--reversal", action="store_true", help="额外输出换向过冲统计（判据②）")
    ap.add_argument("--win-s", type=float, default=0.25, help="换向窗口 ±s")
    args = ap.parse_args()

    session = Path(args.session)
    segs = load(session)
    rep = {}
    for g in sorted(segs):
        d = segs[g]
        t = d[:, 0]
        m = t >= args.t_min
        rep[g] = {}
        for ax, iq, ic in (("fb", 1, 3), ("lr", 2, 4)):
            tau, cc = best_delay(t[m], d[m, iq], d[m, ic])
            e = {"delay_ms": round(tau * 1000, 1), "corr": round(cc, 4),
                 "mae": round(float(np.abs(d[m, iq] - d[m, ic]).mean()), 4)}
            if args.reversal:
                n_rev, mean_signed, mean_abs, max_abs = reversal_overshoot(
                    t[m], d[m, iq], d[m, ic], args.win_s)
                e["reversal"] = {"n": n_rev, "mean_signed_deg": round(mean_signed, 3),
                                 "mean_abs_deg": round(mean_abs, 3), "max_abs_deg": round(max_abs, 3)}
            rep[g][ax] = e
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        for g in sorted(rep):
            line = "seg%d: " % g
            for ax, v in rep[g].items():
                line += "%s 延迟 %6.1f ms / MAE %.3f" % (ax, v["delay_ms"], v["mae"])
                if "reversal" in v:
                    r = v["reversal"]
                    line += " / 换向过冲 n=%d 均值%+.2f° |峰|均值%.2f° 最大%.2f°" % (
                        r["n"], r["mean_signed_deg"], r["mean_abs_deg"], r["max_abs_deg"])
                line += "   "
            print(line)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("matplotlib 不可用，跳过绘图: %s" % e)
        return 0

    # 取每段滞后最大的轴画"对齐前后"
    axes_to_plot = []
    for g in sorted(segs):
        ax = max(("fb", "lr"), key=lambda a: abs(rep[g][a]["delay_ms"]))
        axes_to_plot.append((g, ax))

    fig, axs = plt.subplots(1, len(axes_to_plot), figsize=(7 * len(axes_to_plot), 4.6), squeeze=False)
    for col, (g, axn) in enumerate(axes_to_plot):
        d = segs[g]
        t = d[:, 0]
        iq, ic = (1, 3) if axn == "fb" else (2, 4)
        m = t >= args.t_min
        tau = rep[g][axn]["delay_ms"] / 1000.0
        a = axs[0][col]
        a.plot(t[m], d[m, iq], "--", color="black", lw=1.4, label="target")
        a.plot(t[m], d[m, ic], color="tab:red", lw=1.0, alpha=0.85, label="actual (raw)")
        ts, qs = shift_by(t[m], d[m, ic], tau)
        a.plot(ts, qs, color="tab:green", lw=1.0, ls=":", alpha=0.95,
               label="actual shifted by %.0f ms" % (tau * 1000))
        a.set_title("seg%d %s  --  delay %.0f ms" % (g, axn, tau * 1000))
        a.set_xlabel("t (s)")
        a.set_ylabel("%s (deg)" % axn)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")
    fig.suptitle("Phase lag = pure time delay: after shifting, actual overlaps target", fontsize=12)
    fig.tight_layout()
    out = Path(args.out) if args.out else (session / "phase_lag.png")
    fig.savefig(out, dpi=110)
    print("已写: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
