"""从开环会话测割线增益（滚雪球 40° 扩展用，数据驱动，不猜增益爬升）。

增益随幅度爬升（旧 rig ±1°→±5° 已 1.6×），用小幅增益换算大角度 offset 会超调、
逼近 guardian 70°。本脚本读一个开环会话，按 segment 对每轴拟合过原点斜率
g = Σ(q·u)/Σ(u²)（该幅度下的割线增益），并输出推荐用于下一档幅度的 --gain-fb/--gain-lr。

用法：
    python -m model.measure_gain --session <开环会话目录或servo_data.csv>
    # 输出每段每轴割线增益 + 取"最大幅度段"的推荐覆盖值
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from model.dataset import load_session


def secant_gain(q: np.ndarray, u: np.ndarray) -> float:
    """过原点最小二乘割线增益 g = Σ(q·u)/Σ(u²)。"""
    denom = float((u ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float((q * u).sum() / denom)


def main() -> int:
    ap = argparse.ArgumentParser(description="测开环会话每段割线增益")
    ap.add_argument("--session", nargs="+", required=True,
                    help="会话目录或 servo_data.csv（可多个，合并）")
    args = ap.parse_args()

    qs, us, segs = [], [], []
    for s in args.session:
        p = Path(s)
        csv_path = p / "servo_data.csv" if p.is_dir() else p
        if not csv_path.exists():
            print(f"文件不存在: {csv_path}", file=sys.stderr)
            return 1
        q, u, seg = load_session(csv_path)
        qs.append(q)
        us.append(u)
        segs.append(seg + len(qs) * 1000)
    q = np.concatenate(qs)
    u = np.concatenate(us)
    seg = np.concatenate(segs)

    print("=== 每段割线增益（°/offset，过原点斜率）===")
    print(f"{'seg':>5} {'fb_gain':>10} {'lr_gain':>10} {'fb峰值|u|':>10} {'lr峰值|u|':>10}")
    max_seg = None
    max_amp = -1.0
    for gid in sorted(np.unique(seg)):
        m = seg == gid
        g_fb = secant_gain(q[m, 0], u[m, 0])
        g_lr = secant_gain(q[m, 1], u[m, 1])
        amp = max(np.abs(u[m, 0]).max(), np.abs(u[m, 1]).max())
        print(f"{gid:>5} {g_fb:>10.5f} {g_lr:>10.5f} {np.abs(u[m,0]).max():>10.0f} {np.abs(u[m,1]).max():>10.0f}")
        if amp > max_amp:
            max_amp = amp
            max_seg = (gid, g_fb, g_lr)

    if max_seg is not None:
        print(f"\n推荐（最大幅度段 seg{max_seg[0]}）用于下一档滚雪球的增益覆盖：")
        print(f"  --gain-fb {max_seg[1]:.5f} --gain-lr {max_seg[2]:.5f}")
        print("  ⚠️ 若下一档幅度更大，实际增益可能继续爬升；建议再乘 1.1~1.2 安全系数"
              "（增益↑→offset↓→更安全，宁可少走不可超调）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
