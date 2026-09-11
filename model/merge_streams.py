"""§8.4 D0：四路数据（mocap / servo / imu / force）时间戳对齐 merge 工具。

各采集器独立线程、独立频率（servo/mocap 100Hz、force ~60Hz、imu 100Hz），
落盘为四个 CSV。训练多模态模型（IMU 角速度、张力特征）必须先对齐到统一时间基。

**基准 = servo 时间基**（闭环控制拍的权威时钟）；其余三路取最近邻对齐，
并记录每路的最大时间偏差（超过 --max-gap-ms 的样本标记为无效，避免用陈旧值）。

用法：
    # 单会话
    python -m model.merge_streams --session <会话目录> --out aligned.npz
    # 多会话（拼接，附带 session_id 以便按会话留出）
    python -m model.merge_streams --session A B C --out all.npz
    # 顺便导出 CSV 便于肉眼检查
    python -m model.merge_streams --session A --out a.npz --csv a.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

STREAMS = {
    "mocap": ("mocap_data.csv", ["rigid_body_qx", "rigid_body_qy", "rigid_body_qz", "rigid_body_qw",
                                 "rigid_body_x", "rigid_body_y", "rigid_body_z"]),
    "imu": ("imu_data.csv", ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
                             "ax_no_g_mps2", "ay_no_g_mps2", "az_no_g_mps2",
                             "quat_w", "quat_x", "quat_y", "quat_z"]),
    "force": ("force_data.csv", ["ch1", "ch2", "ch3", "ch4"]),
}
SERVO_COLS = ["t_s", "target_front_back_deg", "target_left_right_deg",
              "current_front_back_deg", "current_left_right_deg",
              "servo_front_back_offset", "servo_left_right_offset", "segment_id"]


def _read(path: Path, cols: list[str]):
    """→ (ts_ms (n,), values (n,k))。缺文件返回 (None, None)。"""
    if not path.exists():
        return None, None
    ts, vals = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ts.append(int(r["pc_receive_unix_time_ms"]))
                vals.append([float(r[c]) for c in cols])
            except (KeyError, ValueError):
                continue
    if not ts:
        return None, None
    ts = np.asarray(ts)
    o = np.argsort(ts)
    return ts[o], np.asarray(vals)[o]


def align_one(session: Path, max_gap_ms: float, servo_baud: int | None = None):
    """把一个会话对齐到 servo 时间基。→ dict 或 None。"""
    sts, sv = _read(session / "servo_data.csv", SERVO_COLS)
    if sts is None:
        return None
    out = {
        "t_ms": sts.astype(np.int64),
        "servo": sv,
        "segment_id": sv[:, SERVO_COLS.index("segment_id")].astype(np.int64),
    }
    gaps = {}
    for name, (fname, cols) in STREAMS.items():
        ts, vals = _read(session / fname, cols)
        if ts is None:
            out[name] = np.full((len(sts), len(cols)), np.nan)
            out[f"{name}_valid"] = np.zeros(len(sts), dtype=bool)
            gaps[name] = float("nan")
            continue
        # 最近邻
        idx = np.clip(np.searchsorted(ts, sts), 0, len(ts) - 1)
        left = np.clip(idx - 1, 0, len(ts) - 1)
        pick = np.where(np.abs(ts[left] - sts) <= np.abs(ts[idx] - sts), left, idx)
        gap = np.abs(ts[pick] - sts)
        out[name] = vals[pick]
        out[f"{name}_valid"] = gap <= max_gap_ms
        gaps[name] = float(np.median(gap))
    return out, gaps


def main() -> int:
    ap = argparse.ArgumentParser(description="§8.4 D0 四路时间戳对齐 merge")
    ap.add_argument("--session", nargs="+", required=True, help="会话目录（可多个）")
    ap.add_argument("--out", required=True, help="输出 .npz")
    ap.add_argument("--csv", default=None, help="可选：同时导出 CSV（便于肉眼检查）")
    ap.add_argument("--max-gap-ms", type=float, default=120.0,
                    help="允许的最大对齐时间偏差（超过标记 invalid）")
    args = ap.parse_args()

    merged = {}
    sess_arrays = []
    for i, s in enumerate(args.session):
        p = Path(s)
        p = p if p.is_dir() else p.parent
        r = align_one(p, args.max_gap_ms)
        if r is None:
            print(f"跳过（缺 servo_data.csv）: {p}", file=sys.stderr)
            continue
        d, gaps = r
        n = len(d["t_ms"])
        print(f"{p.name}: {n} 拍  对齐偏差(中位) "
              + "  ".join(f"{k}={v:.1f}ms" for k, v in gaps.items()))
        d["session_id"] = np.full(n, i, dtype=np.int64)
        sess_arrays.append(d)

    if not sess_arrays:
        print("无有效会话", file=sys.stderr)
        return 1

    keys = sess_arrays[0].keys()
    for k in keys:
        merged[k] = np.concatenate([d[k] for d in sess_arrays])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)
    print(f"\n已写: {out}  总拍数 {len(merged['t_ms'])}  字段: {sorted(merged.keys())}")

    if args.csv:
        cols = ["t_ms", "session_id", "segment_id"]
        cols += [f"servo_{c}" for c in SERVO_COLS]
        for name, (_, cs) in STREAMS.items():
            cols += [f"{name}_{c}" for c in cs] + [f"{name}_valid"]
        cp = Path(args.csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in range(len(merged["t_ms"])):
                row = [merged["t_ms"][i], merged["session_id"][i], merged["segment_id"][i]]
                row += list(merged["servo"][i])
                for name in STREAMS:
                    row += list(merged[name][i]) + [int(merged[f"{name}_valid"][i])]
                w.writerow(row)
        print(f"已写 CSV: {cp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
