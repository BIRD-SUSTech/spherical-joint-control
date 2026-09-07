"""残差结构分解（设计文档 §7.3 四维标签法，M5 核心产出）。

对留出段算 1-step 残差 e = q_m − q̂，按四个物理维度分组统计：
    1. 方向：sign(u) → 正/负
    2. 换向：since_reversal → 刚换向/过渡/稳定
    3. 速度：|q̇| 分位数 → 低/中/高
    4. 幅度：|u| → 小/中/大

每组统计 mean|e| / RMSE / max|e|，残差大的组 = 固有问题所在。
"""

from __future__ import annotations

import numpy as np


def _stats(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {"n": 0, "mae": 0.0, "rmse": 0.0, "max": 0.0}
    return {
        "n": int(len(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max": float(np.max(np.abs(err))),
    }


def _bins_by_percentile(vals: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """按分位数分箱，返回 0..n_bins-1 的箱号。"""
    qs = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(vals, qs)


def decompose(e: np.ndarray, q: np.ndarray, u: np.ndarray,
              qdot: np.ndarray, since_rev: np.ndarray) -> dict:
    """四维残差分组。e/q/u 形状 (n,2)。

    返回 { "<axis>_<dim>_<group>": stats, ... }
    """
    results = {}
    axes = ["fb", "lr"]

    for a, name in enumerate(axes):
        r = e[:, a]
        su = np.sign(u[:, a])
        sr = since_rev[:, a]
        speed = np.abs(qdot[:, a])
        amp = np.abs(u[:, a])

        # 维度1：方向
        results[f"{name}_dir_pos"] = _stats(r[su > 0])
        results[f"{name}_dir_neg"] = _stats(r[su < 0])

        # 维度2：换向（since_reversal 分箱）
        rev_bin = np.digitize(sr, [0.1, 0.5])  # 0=刚换向, 1=过渡, 2=稳定
        for b, label in enumerate(["just_rev", "transient", "steady"]):
            results[f"{name}_rev_{label}"] = _stats(r[rev_bin == b])

        # 维度3：速度（|q̇| 分位数）
        sp_bin = _bins_by_percentile(speed, 3)
        for b, label in enumerate(["slow", "mid", "fast"]):
            results[f"{name}_speed_{label}"] = _stats(r[sp_bin == b])

        # 维度4：幅度（|u| 分箱：≤28 小 / ≤56 中 / >56 大）
        amp_bin = np.digitize(amp, [28.0, 56.0])
        for b, label in enumerate(["small", "mid", "large"]):
            results[f"{name}_amp_{label}"] = _stats(r[amp_bin == b])

    return results


def format_report(results: dict, axis: str = "fb") -> str:
    """格式化单轴残差分解报告。"""
    lines = [f"=== {axis} 残差分解（mean|e|° / RMSE / max / n）==="]
    for dim, groups in [
        ("方向", ["dir_pos", "dir_neg"]),
        ("换向", ["rev_just_rev", "rev_transient", "rev_steady"]),
        ("速度", ["speed_slow", "speed_mid", "speed_fast"]),
        ("幅度", ["amp_small", "amp_mid", "amp_large"]),
    ]:
        for g in groups:
            key = f"{axis}_{g}"
            s = results[key]
            lines.append(f"  {dim:4s} {g:12s}  mae={s['mae']:.4f}  rmse={s['rmse']:.4f}  "
                         f"max={s['max']:.4f}  n={s['n']}")
    return "\n".join(lines)
