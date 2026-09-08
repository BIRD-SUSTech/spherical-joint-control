"""拟合控制器前馈参数（参数化逆映射 g(q)）→ controller JSON。

两种数据源：
    - 开环（默认）：从开环 (u, q) 拟合，是稳态的"近似"（被动态滞后污染）。
    - 收敛段（--converged-only）：从闭环收敛段 (q_d, u_log) 拟合，是稳态的"真值"
      （g 机制 A：u_log = 让球杆停在 q_d 的真实 offset）。

用法：
    # 开环拟合
    python -m model.fit_controller --session <A> <B> --out configs/controller.json
    # 收敛段拟合（g 机制 A，推荐）
    python -m model.fit_controller --converged-only --session <A> <B> \
        --out configs/controller_v5.json --slew-limit 2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from model.dataset import load_session
from model.gain_schedule import fit_inverse_poly


def load_converged(csv_paths, threshold_deg: float = 0.3):
    """从闭环会话提取收敛段 (q_d, u_log)。

    收敛段：|current − target| < threshold（两轴都满足），
    此时 u_log = servo offset ≈ 让球杆停在 q_d 的真实稳态指令。
    """
    qs, us = [], []
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("segment_id", "")
                if sid in ("", "-1", None):
                    continue
                try:
                    tf = float(row["target_front_back_deg"])
                    cf = float(row["current_front_back_deg"])
                    tl = float(row["target_left_right_deg"])
                    cl = float(row["current_left_right_deg"])
                    uf = float(row["servo_front_back_offset"])
                    ul = float(row["servo_left_right_offset"])
                except (KeyError, ValueError):
                    continue
                if abs(cf - tf) < threshold_deg and abs(cl - tl) < threshold_deg:
                    qs.append([tf, tl])
                    us.append([uf, ul])
    return np.array(qs), np.array(us)


def main() -> int:
    ap = argparse.ArgumentParser(description="拟合参数化前馈（逆映射 g(q)）")
    ap.add_argument("--session", nargs="+", required=True,
                    help="会话目录或 servo_data.csv（可多个）")
    ap.add_argument("--out", default="configs/controller.json", help="输出 controller JSON")
    ap.add_argument("--degree", type=int, default=3, help="逆映射多项式阶数")
    ap.add_argument("--converged-only", action="store_true",
                    help="只用闭环收敛段 (q_d, u_log) 拟合（g 机制 A）")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="收敛阈值（度），|current-target| 小于它才计入收敛段")
    ap.add_argument("--slew-limit", type=float, default=None, help="slew 限幅（可选写入）")
    args = ap.parse_args()

    csv_paths = []
    for s in args.session:
        p = Path(s)
        csv_path = p / "servo_data.csv" if p.is_dir() else p
        if not csv_path.exists():
            print(f"文件不存在: {csv_path}", file=sys.stderr)
            return 1
        csv_paths.append(csv_path)

    if args.converged_only:
        q, u = load_converged(csv_paths, args.threshold)
        print(f"收敛段样本: {len(q)}（阈值 {args.threshold}°）")
        if len(q) < 30:
            print("收敛段样本太少，无法拟合", file=sys.stderr)
            return 1
    else:
        qs, us = [], []
        for csv_path in csv_paths:
            qq, uu, _ = load_session(csv_path)
            qs.append(qq)
            us.append(uu)
        q = np.concatenate(qs)
        u = np.concatenate(us)
        print(f"开环样本: {len(q)}")

    gain_poly = {}
    for ax, name in enumerate(["fb", "lr"]):
        coeffs = fit_inverse_poly(q[:, ax], u[:, ax], degree=args.degree)
        gain_poly[name] = [float(c) for c in coeffs]
        print(f"{name}: " + "  ".join(f"{c:+.6f}" for c in coeffs))

    data: dict = {"gain_poly": gain_poly}
    if args.slew_limit is not None:
        data["slew_limit"] = args.slew_limit

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
