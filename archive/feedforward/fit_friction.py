"""从开环数据拟合结构化摩擦前馈参数 G, f_c, f_s。

每轴模型：
    d_ff = G·Δq + f_c·sign(Δq) + f_s·sign(Δq)·起步脉冲
    |Δq| < deadband 时 d_ff = 0

拟合（稳健，避免用噪声大的每拍增量直接拟合逆映射）：
    G   ：位置-差分前向拟合 q ≈ a·d + b 的斜率倒数  G = 1/a
          （等价于准静态"多少差分产生多少位置变化"，R² 高）
    f_s ：按 |d| 分桶，找 median|Δq| 越过噪声地板的最小 |d|（死区边界）
    f_c ：取 f_c = ratio·f_s（典型 Coulomb/static 比 ~0.7）
    deadband = f_s / |G|（静摩擦对应的等效 Δq，可覆盖）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feedforward.dataset import load_servo_df


def fit_g(q, d):
    """位置-差分前向拟合 q ≈ a·d + b，返回 G=1/a 和 R²。"""
    a, b = np.polyfit(d, q, 1)
    pred = a * d + b
    r2 = 1 - ((q - pred) ** 2).sum() / ((q - q.mean()) ** 2).sum()
    return 1.0 / a, r2


def estimate_fs(d, dq, noise_floor=0.05, nbins=40):
    """按 |d| 分桶，找 median|Δq| 越过噪声地板的最小 |d|（死区边界）。"""
    absd = np.abs(d)
    edges = np.linspace(absd.min(), absd.max(), nbins + 1)
    centers, meds = [], []
    for i in range(nbins):
        m = (absd >= edges[i]) & (absd < edges[i + 1])
        if m.sum() > 30:
            centers.append((edges[i] + edges[i + 1]) / 2)
            meds.append(np.median(np.abs(dq[m])))
    centers = np.array(centers)
    meds = np.array(meds)
    idx = np.where(meds > noise_floor)[0]
    return float(centers[idx[0]]) if len(idx) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--noise-floor", type=float, default=0.05, help="死区判定噪声地板 (°/拍)")
    ap.add_argument("--coulomb-ratio", type=float, default=0.7, help="f_c/f_s 比")
    ap.add_argument("--deadband", type=float, default=0.02, help="前馈死区 (°/拍)，|Δq|<死区时不输出")
    ap.add_argument("--pretension", type=float, default=0.15)
    ap.add_argument("--out", default="feedforward/outputs/friction_config.json")
    args = ap.parse_args()

    df = load_servo_df(Path(args.data_dir) / "servo_data.csv")
    df = df[df["segment_id"] >= 0].reset_index(drop=True)
    q = df[["current_pitch", "current_yaw"]].to_numpy(float)
    u = df[["servo_1_target_deg", "servo_2_target_deg",
            "servo_3_target_deg", "servo_4_target_deg"]].to_numpy(float)
    ks = np.arange(0, len(q) - 1)
    dq = q[ks + 1] - q[ks]                 # (M,2) 每拍增量
    d = u[ks, 0:2] - u[ks, 2:4]            # (M,2) 差分 [u1-u3, u2-u4]

    out = {"pretension": args.pretension, "noise_floor": args.noise_floor}
    for i, name in enumerate(["pitch", "yaw"]):
        G, r2 = fit_g(q[ks, i], d[:, i])
        fs = estimate_fs(d[:, i], dq[:, i], noise_floor=args.noise_floor)
        fc = args.coulomb_ratio * fs
        motion_thresh = (fs - fc) / abs(G)
        out[f"{name}_G"] = round(float(G), 6)
        out[f"{name}_f_c"] = round(float(fc), 6)
        out[f"{name}_f_s"] = round(float(fs), 6)
        out[f"{name}_deadband_deg"] = args.deadband
        print(f"{name}: G={G:+.5f} norm/°  f_s={fs:.4f}  f_c={fc:.4f}  "
              f"起跳阈值={motion_thresh:.4f}°/拍  deadband={args.deadband}°/拍  (R²={r2:.4f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
