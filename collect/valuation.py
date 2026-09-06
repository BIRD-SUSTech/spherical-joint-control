"""闭环跟踪误差评估：读统一 schema 的 servo_data.csv。

target 与 current 已落在同一行（M2 schema），无需跨文件对齐。
按轴计算 MAE / RMSE / 最大绝对误差，只统计 segment_id >= 0 的数据段。

用法：
    python -m collect.valuation <session_dir>          # 自动找 servo_data.csv
    python -m collect.valuation <servo_data.csv>       # 直接给 CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _stats(errors: list[float]) -> dict:
    if not errors:
        return {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0, "n": 0}
    n = len(errors)
    mae = sum(abs(e) for e in errors) / n
    rmse = (sum(e * e for e in errors) / n) ** 0.5
    mx = max(abs(e) for e in errors)
    return {"mae": mae, "rmse": rmse, "max_abs": mx, "n": n}


def evaluate(servo_csv: Path) -> dict:
    fb_errs: list[float] = []
    lr_errs: list[float] = []

    with open(servo_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("segment_id", "")
            if sid in ("", "-1", None):  # 只统计数据段
                continue
            try:
                t_fb = float(row["target_front_back_deg"])
                t_lr = float(row["target_left_right_deg"])
                c_fb = float(row["current_front_back_deg"])
                c_lr = float(row["current_left_right_deg"])
            except (KeyError, ValueError):
                continue
            fb_errs.append(t_fb - c_fb)
            lr_errs.append(t_lr - c_lr)

    return {
        "front_back": _stats(fb_errs),
        "left_right": _stats(lr_errs),
    }


def _resolve_csv(path: Path) -> Path:
    if path.is_dir():
        return path / "servo_data.csv"
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="闭环跟踪误差评估")
    ap.add_argument("path", help="会话目录或 servo_data.csv")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    csv_path = _resolve_csv(Path(args.path))
    if not csv_path.exists():
        print(f"文件不存在: {csv_path}", file=sys.stderr)
        return 1

    result = evaluate(csv_path)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for axis, label in (("front_back", "前后"), ("left_right", "左右")):
            s = result[axis]
            print(f"{label}: MAE={s['mae']:.4f}°  RMSE={s['rmse']:.4f}°  "
                  f"max|e|={s['max_abs']:.4f}°  n={s['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
