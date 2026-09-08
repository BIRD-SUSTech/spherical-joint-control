"""独立评测：误差评估 + 可视化 + 结果落盘。

读 servo_data.csv（target 与 current 同行，M2 schema），按 segment_id 分段：
评估（MAE/RMSE/max）+ 可视化（两轴时间图 + 俯视角轨迹 + 动图 GIF）。

用法：
    python -m valuation.evaluate <session> --all          # 全部评测（metric + 图 + gif）
    python -m valuation.evaluate <session> --ab           # A/B 分段对比
    python -m valuation.evaluate <session> --plot time    # 单项图
    python -m valuation.evaluate <session> --plot animate --gif out.gif

结果输出（--all 时）到 valuation/results/<session名>/：
    metrics.json / track_time.png / track_top.png / track_anim.gif
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RESULT_ROOT = Path(__file__).resolve().parent / "results"


def _stats(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0, "n": 0}
    return {
        "n": int(len(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def load_trajectory(servo_csv: Path, segment: int | None = None):
    """读轨迹 → (t, target_fb, target_lr, current_fb, current_lr)。segment=None 用全部数据段。"""
    t, tf, tl, cf, cl = [], [], [], [], []
    with open(servo_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("segment_id", "")
            if sid in ("", "-1", None):
                continue
            if segment is not None and int(sid) != segment:
                continue
            try:
                t.append(float(row["t_s"]))
                tf.append(float(row["target_front_back_deg"]))
                tl.append(float(row["target_left_right_deg"]))
                cf.append(float(row["current_front_back_deg"]))
                cl.append(float(row["current_left_right_deg"]))
            except (KeyError, ValueError):
                continue
    return (np.array(t), np.array(tf), np.array(tl), np.array(cf), np.array(cl))


def evaluate(servo_csv: Path, segment: int | None = None) -> dict:
    """按段（或全部）评估前后/左右跟踪误差。"""
    _, tf, tl, cf, cl = load_trajectory(servo_csv, segment)
    return {
        "front_back": _stats(tf - cf),
        "left_right": _stats(tl - cl),
    }


def _segments(servo_csv: Path) -> list[int]:
    segs = set()
    with open(servo_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("segment_id", "")
            if sid not in ("", "-1", None):
                segs.add(int(sid))
    return sorted(segs)


def _resolve_csv(path: Path) -> Path:
    return path / "servo_data.csv" if path.is_dir() else path


def _session_name(servo_csv: Path) -> str:
    """从会话目录/CSV 路径提取会话名。"""
    return servo_csv.parent.name if servo_csv.name == "servo_data.csv" else servo_csv.stem


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _plot_backend():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_time_series(servo_csv: Path, segment: int | None = None, out: Path | None = None,
                     label: str = "") -> Path:
    """每个 segment 一个子图（时间轴各自从 0 起），fb/lr × target/current 四线。"""
    plt = _plot_backend()
    segs = _segments(servo_csv) if segment is None else [segment]
    fig, axes = plt.subplots(len(segs), 1, figsize=(11, 2.6 * max(len(segs), 1)),
                             sharex=False, squeeze=False)
    axes = axes[:, 0]
    for ax, seg in zip(axes, segs):
        t, tf, tl, cf, cl = load_trajectory(servo_csv, seg)
        t = t - t[0]  # 每段时间轴从 0 起
        ax.plot(t, tf, "--", color="blue", lw=1.2, label="fb target", alpha=0.9)
        ax.plot(t, cf, color="blue", lw=1.0, label="fb current", alpha=0.75)
        ax.plot(t, tl, "--", color="red", lw=1.2, label="lr target", alpha=0.9)
        ax.plot(t, cl, color="red", lw=1.0, label="lr current", alpha=0.75)
        ax.set_ylabel(f"seg{seg} (deg)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"Tracking per segment: {label}")
    fig.tight_layout()
    png = out or (servo_csv.parent / "track_time.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def plot_trajectory(servo_csv: Path, segment: int | None = None, out: Path | None = None,
                    label: str = "", ax_lim: float | None = None) -> Path:
    plt = _plot_backend()
    _, tf, tl, cf, cl = load_trajectory(servo_csv, segment)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(tf, tl, "--", lw=1.5, color="blue", label="target", alpha=0.9)
    ax.plot(cf, cl, lw=1.2, color="orange", label="current", alpha=0.8)
    ax.scatter([tf[0]], [tl[0]], color="blue", s=60, label="start", zorder=5)
    ax.set_xlabel("front-back (deg)")
    ax.set_ylabel("left-right (deg)")
    ax.set_title(f"Top-view: {label or 'segment ' + str(segment)}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    if ax_lim is not None:
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
    fig.tight_layout()
    png = out or (servo_csv.parent / "track_top.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def animate_trajectory(servo_csv: Path, segment: int | None = None,
                       gif: Path | None = None, label: str = "") -> Path:
    plt = _plot_backend()
    import matplotlib.animation as animation

    _, tf, tl, cf, cl = load_trajectory(servo_csv, segment)
    x_exp, y_exp = tf, tl
    x_act, y_act = cf, cl

    fig, ax = plt.subplots(figsize=(8, 8))
    x_min = min(x_exp.min(), x_act.min())
    x_max = max(x_exp.max(), x_act.max())
    y_min = min(y_exp.min(), y_act.min())
    y_max = max(y_exp.max(), y_act.max())
    pad_x = (x_max - x_min) * 0.1 or 1
    pad_y = (y_max - y_min) * 0.1 or 1
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    (line_exp,) = ax.plot([], [], "--", color="blue", lw=1.5, label="target", alpha=0.9)
    (line_act,) = ax.plot([], [], color="orange", lw=1.2, label="current", alpha=0.8)
    (head_exp,) = ax.plot([], [], "bo", markersize=7)
    (head_act,) = ax.plot([], [], "o", color="orange", markersize=7)

    ax.set_xlabel("front-back (deg)")
    ax.set_ylabel("left-right (deg)")
    ax.set_title(f"Top-view animation: {label or 'segment ' + str(segment)}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    step = max(1, len(x_exp) // 300)

    def update(frame):
        idx = min(frame, len(x_exp) - 1)
        line_exp.set_data(x_exp[:idx + 1], y_exp[:idx + 1])
        line_act.set_data(x_act[:idx + 1], y_act[:idx + 1])
        head_exp.set_data([x_exp[idx]], [y_exp[idx]])
        head_act.set_data([x_act[idx]], [y_act[idx]])
        return line_exp, line_act, head_exp, head_act

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(x_exp), step),
                                  interval=20, blit=True, repeat=False)
    gif_path = gif or (servo_csv.parent / "track_anim.gif")
    try:
        ani.save(gif_path, writer="pillow", fps=30)
    except Exception as e:  # noqa: BLE001
        plt.close(fig)
        raise RuntimeError(f"GIF 保存失败（需 pillow）: {e}")
    plt.close(fig)
    return gif_path


# ---------------------------------------------------------------------------
# 全部评测 + 结果落盘
# ---------------------------------------------------------------------------

def _build_metrics(servo_csv: Path) -> dict:
    """构造指标 dict（含分段评估 + A/B 变化对比）。"""
    segs = _segments(servo_csv)
    metrics: dict = {"segments": {str(s): evaluate(servo_csv, s) for s in segs}}
    if len(segs) >= 2:
        r0 = metrics["segments"][str(segs[0])]
        r1 = metrics["segments"][str(segs[-1])]
        metrics["comparison"] = {}
        for axis, label in (("front_back", "前后"), ("left_right", "左右")):
            d_mae = (r1[axis]["mae"] - r0[axis]["mae"]) / r0[axis]["mae"] * 100
            d_max = (r1[axis]["max_abs"] - r0[axis]["max_abs"]) / r0[axis]["max_abs"] * 100
            metrics["comparison"][label] = {"mae_pct": round(d_mae, 2), "max_pct": round(d_max, 2)}
    return metrics


def run_all(servo_csv: Path) -> Path:
    """一次全部评测 → valuation/results/<session名>/（metric + 图 + gif）。"""
    session = _session_name(servo_csv)
    out_dir = RESULT_ROOT / session
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _build_metrics(servo_csv)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_time_series(servo_csv, out=out_dir / "track_time.png")
    plot_trajectory(servo_csv, out=out_dir / "track_top.png")
    animate_trajectory(servo_csv, gif=out_dir / "track_anim.gif")
    return out_dir


def _print_ab(servo_csv: Path) -> None:
    segs = _segments(servo_csv)
    print("=== A/B 分段评估 ===")
    for s in segs:
        r = evaluate(servo_csv, s)
        fb, lr = r["front_back"], r["left_right"]
        print(f"  segment {s}: fb MAE={fb['mae']:.4f} RMSE={fb['rmse']:.4f} max={fb['max_abs']:.4f}"
              f"  |  lr MAE={lr['mae']:.4f} RMSE={lr['rmse']:.4f} max={lr['max_abs']:.4f}")
    if len(segs) >= 2:
        m = _build_metrics(servo_csv)["comparison"]
        print("\n--- 末段 vs 首段 变化 ---")
        for label, d in m.items():
            print(f"  {label}: MAE {d['mae_pct']:+.1f}%  max {d['max_pct']:+.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="独立评测：误差评估 + 可视化 + 结果落盘")
    ap.add_argument("path", help="会话目录或 servo_data.csv")
    ap.add_argument("--all", action="store_true", help="一次全部评测（metric+图+gif）")
    ap.add_argument("--segment", type=int, default=None, help="只评估/可视化指定段 id")
    ap.add_argument("--ab", action="store_true", help="A/B 分段对比（所有段）")
    ap.add_argument("--plot", choices=("time", "top", "animate"), default=None,
                    help="可视化：time=两轴时间图 top=俯视角 animate=动图")
    ap.add_argument("--gif", default=None, help="--plot animate 的 GIF 输出路径")
    ap.add_argument("--out", default=None, help="PNG/GIF 输出路径")
    ap.add_argument("--json", action="store_true", help="评估结果输出 JSON")
    args = ap.parse_args()

    csv_path = _resolve_csv(Path(args.path))
    if not csv_path.exists():
        print(f"文件不存在: {csv_path}", file=sys.stderr)
        return 1

    if args.all:
        out_dir = run_all(csv_path)
        print(f"全部评测完成，结果目录: {out_dir}")
        _print_ab(csv_path)
        return 0

    if args.ab:
        _print_ab(csv_path)
        return 0

    result = evaluate(csv_path, args.segment)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for axis, label in (("front_back", "前后"), ("left_right", "左右")):
            s = result[axis]
            print(f"{label}: MAE={s['mae']:.4f}°  RMSE={s['rmse']:.4f}°  "
                  f"max|e|={s['max_abs']:.4f}°  n={s['n']}")

    if args.plot:
        out = Path(args.out) if args.out else None
        seg_label = f"seg{args.segment}" if args.segment is not None else "all"
        if args.plot == "time":
            p = plot_time_series(csv_path, args.segment, out, seg_label)
        elif args.plot == "top":
            p = plot_trajectory(csv_path, args.segment, out, seg_label)
        else:
            gif = Path(args.gif) if args.gif else out
            p = animate_trajectory(csv_path, args.segment, gif, seg_label)
        print(f"已保存: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
