"""D0 重采验收检查：标签是否干净（限幅残留 + 方差分解）。

背景（D2v2 否决报告 §5/§6）：
  旧 D0 数据在 `slew_limit=2.0`（200 offset/s）下采集，快轴前馈被限幅：
    - 前馈速率需求 592–993 offset/s（p99），18–60% 采样点触限
    - 训练标签 r = u_总 − g_static(q_d) 与"纯限幅缺口"相关 0.30–0.81
    - 且真实速度项与限幅缺口**数学上共线**（缺口 ≈ −τ·g′·q̇）→ 无法分离
  本工具给出重采后的三项验收指标：

  ① 限幅残留：按【当前基线配置】的 slew_limit 计算需求速率超限占比（目标 0%）
  ② 抖动/退化：label rms、r/std(e)、corr(r, gap)（目标 corr → ~0）
  ③ 方差分解：标准化 r ~ [gap, q̇, q̈] 的系数与 R²（用于判断主导动态项是速度型还是加速度型）

用法：
    python -m model.check_d0_labels collect/logs/session_xxx [session_yyy ...]
    python -m model.check_d0_labels <会话...> --base-config configs/static_feedforward_controller.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.controller_config import ControllerConfig, _poly  # noqa: E402

LOOP_HZ = 100.0          # control/loop.py: LOOP_HZ
THRESHOLD = 1.0          # 收敛段阈值（°，与 fit_dynamic_nn.py 默认一致）


def _load_servo(session: Path) -> dict[int, np.ndarray]:
    cols = ("t_s", "target_front_back_deg", "target_left_right_deg",
            "current_front_back_deg", "current_left_right_deg",
            "servo_front_back_offset", "servo_left_right_offset")
    segs: dict[int, list] = {}
    with (session / "servo_data.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                sid = int(row["segment_id"])
            except (KeyError, ValueError):
                continue
            if sid < 0:
                continue
            try:
                segs.setdefault(sid, []).append(tuple(float(row[c]) for c in cols))
            except (KeyError, ValueError):
                continue
    return {k: np.array(sorted(v)) for k, v in segs.items()}


def _slew_sim(u: np.ndarray, limit: float) -> np.ndarray:
    out = np.empty_like(u)
    prev = u[0]
    for i, cur in enumerate(u):
        d = cur - prev
        prev = prev + limit * (1 if d > 0 else -1) if abs(d) > limit else cur
        out[i] = prev
    return out


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description="D0 重采验收：标签干净度检查")
    ap.add_argument("--session", nargs="+", required=True, help="会话目录（可多个，合并评估）")
    ap.add_argument("--base-config", default="configs/static_feedforward_controller.json")
    ap.add_argument("--slew-limit", type=float, default=None,
                    help="覆盖配置里的 slew_limit（检查旧会话时用它还原当时的限幅值，如 2.0）")
    args = ap.parse_args()

    cfg = ControllerConfig.load(args.base_config)
    if cfg.gain_poly is None:
        print("base-config 缺 gain_poly，无法定义 g_static", file=sys.stderr)
        return 2
    slew = cfg.slew_limit if args.slew_limit is None else args.slew_limit
    if args.slew_limit is not None:
        print("（slew_limit 已按 --slew-limit 覆盖为 %s，用于复核历史会话）" % slew)
    print("基线配置: %s" % args.base_config)
    print("  slew_limit=%s offset/拍 @%.0fHz → 上限 %s offset/s"
          % (slew, LOOP_HZ, "不限" if slew is None else "%.0f" % (slew * LOOP_HZ)))
    print()

    pooled: dict[str, list] = {"fb": [], "lr": []}
    hdr = "%-26s %-5s %9s %9s %8s %9s %9s %9s" % (
        "会话", "轴", "需求p99", "需求max", "超限%", "label_rms", "std(e)", "corr(r,gap)")
    print(hdr)
    print("-" * len(hdr))

    for s in args.session:
        session = Path(s)
        if not session.exists():
            print("跳过（不存在）: %s" % session)
            continue
        for sid, d in sorted(_load_servo(session).items()):
            t = d[:, 0]
            m = t >= 5.0
            tf, tl, cf, cl = d[:, 1], d[:, 2], d[:, 3], d[:, 4]
            uf, ul = d[:, 5], d[:, 6]
            conv = m & (np.abs(tf - cf) < THRESHOLD) & (np.abs(tl - cl) < THRESHOLD)
            for ax, q, cur, u in (("fb", tf, cf, uf), ("lr", tl, cl, ul)):
                g = np.array([_poly(cfg.gain_poly[ax], v) for v in q])
                req = np.abs(np.gradient(g, 1.0 / LOOP_HZ))
                gap = np.zeros_like(g) if slew is None else g - _slew_sim(g, slew)
                r = u - g
                over = float((req[m] > slew * LOOP_HZ).mean()) * 100 if slew is not None else 0.0
                cc = (float(np.corrcoef(r[conv], gap[conv])[0, 1])
                      if gap[conv].std() > 1e-9 else 0.0)
                print("%-26s %-5s %9.0f %9.0f %7.1f%% %9.1f %9.3f %9.3f"
                      % (session.name[-11:] + "#s%d" % sid, ax, np.percentile(req[m], 99),
                         req.max(), over, r[conv].std(), (q - cur)[conv].std(), cc))
                if conv.sum() > 50:
                    pooled[ax].append(np.vstack([r[conv], gap[conv],
                                                 np.gradient(q, 1.0 / LOOP_HZ)[conv],
                                                 np.gradient(np.gradient(q, 1.0 / LOOP_HZ),
                                                             1.0 / LOOP_HZ)[conv]]))
        print()

    print("=" * 72)
    print("标签方差分解（标准化 r ~ gap + q̇ + q̈，收敛段 |e|<%.1f°）" % THRESHOLD)
    print("%-6s %7s %11s %9s %9s %10s %10s %10s" %
          ("轴", "n", "gap", "q̇", "q̈", "R²全", "R²仅q̈", "R²仅q̇"))
    print("-" * 76)
    for ax in ("fb", "lr"):
        if not pooled[ax]:
            continue
        A = np.hstack(pooled[ax])
        r, gap, qd, qdd = (_z(v) for v in A)
        X = np.vstack([gap, qd, qdd]).T
        coef, *_ = np.linalg.lstsq(X, r, rcond=None)
        r2 = 1 - ((r - X @ coef) ** 2).sum() / (r ** 2).sum()
        r2_qdd = float(np.corrcoef(qdd, r)[0, 1]) ** 2
        r2_qd = float(np.corrcoef(qd, r)[0, 1]) ** 2
        print("%-6s %7d %11.3f %9.3f %9.3f %10.3f %10.3f %10.3f"
              % (ax, len(r), coef[0], coef[1], coef[2], r2, r2_qdd, r2_qd))

    print()
    print("验收标准：")
    print("  ① 超限% 全为 0（前馈不再被限幅）")
    print("  ② corr(r, gap) → ~0（缺口不再是标签成分）")
    print("  ③ R²仅q̈ 若从 0.11 显著上升 → q̈ 项终于可辨识；若仍低 → 主导项是速度型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
