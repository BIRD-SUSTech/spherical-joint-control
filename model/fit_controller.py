"""拟合控制器前馈参数（参数化逆映射 g(q)）→ controller JSON。

从开环 (u, q) 数据拟合逆映射 u = g(q)（三阶多项式），写 controller JSON。
这是螺旋迭代里"每轮重拟合控制器参数"的工具。

用法：
    python -m model.fit_controller --session <A> <B> --out configs/controller_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from model.dataset import load_session
from model.gain_schedule import fit_inverse_poly


def main() -> int:
    ap = argparse.ArgumentParser(description="拟合参数化前馈（逆映射 g(q)）")
    ap.add_argument("--session", nargs="+", required=True, help="开环会话目录或 servo_data.csv（可多个）")
    ap.add_argument("--out", default="configs/controller_v2.json", help="输出 controller JSON")
    ap.add_argument("--degree", type=int, default=3, help="逆映射多项式阶数")
    args = ap.parse_args()

    qs, us = [], []
    for s in args.session:
        p = Path(s)
        csv_path = p / "servo_data.csv" if p.is_dir() else p
        if not csv_path.exists():
            print(f"文件不存在: {csv_path}", file=sys.stderr)
            return 1
        q, u, _ = load_session(csv_path)
        qs.append(q)
        us.append(u)
    q = np.concatenate(qs)
    u = np.concatenate(us)

    gain_poly = {}
    for ax, name in enumerate(["fb", "lr"]):
        coeffs = fit_inverse_poly(q[:, ax], u[:, ax], degree=args.degree)
        gain_poly[name] = [float(c) for c in coeffs]
        print(f"{name}: " + "  ".join(f"{c:+.6f}" for c in coeffs))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gain_poly": gain_poly}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"已写: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
