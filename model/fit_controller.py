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
    # 收敛段拟合 + 几何耦合交叉项（留一会话护栏，只在不劣于 1D 时写入 gain_cross）
    python -m model.fit_controller --converged-only --couple --session <A> <B> <C> \
        --out configs/controller_v8_candidate.json --slew-limit 2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from model.dataset import load_session
from model.gain_schedule import (eval_additive_cross, eval_poly,
                                 fit_additive_cross, fit_inverse_poly,
                                 inverse_is_monotonic)


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


def _predict_1d(q: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return eval_poly(coeffs, q)


def _predict_cross(q2d: np.ndarray, own: np.ndarray, cross: np.ndarray, ax: int) -> np.ndarray:
    """ax=0(fb): q_self=列0, q_other=列1；ax=1(lr) 反之。"""
    q_self = q2d[:, ax]
    q_other = q2d[:, 1 - ax]
    return eval_additive_cross(own, cross, q_self, q_other)


def _rmse(pred: np.ndarray, u: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - u) ** 2)))


def _loso_compare(sessions, degree: int):
    """留一会话交叉验证：1D vs 加性交叉逆映射的留出 RMSE（offset 单位）。

    sessions: [(q_s, u_s), ...]，每个 q_s/u_s 形状 (n_s,2)。
    返回 (dict 1D, dict cross)，各含 {"fb": rmse, "lr": rmse}（按样本数加权平均）。
    """
    n = len(sessions)
    rmse1 = {"fb": 0.0, "lr": 0.0}
    rmsec = {"fb": 0.0, "lr": 0.0}
    tot = 0
    for k in range(n):
        tr = [sessions[i] for i in range(n) if i != k]
        q_tr = np.concatenate([s[0] for s in tr])
        u_tr = np.concatenate([s[1] for s in tr])
        q_te, u_te = sessions[k]
        tot += len(q_te)

        for ax, name in enumerate(["fb", "lr"]):
            c1 = fit_inverse_poly(q_tr[:, ax], u_tr[:, ax], degree=degree)
            rmse1[name] += len(q_te) * _rmse(_predict_1d(q_te[:, ax], c1), u_te[:, ax])

            own, cross = fit_additive_cross(q_tr[:, ax], q_tr[:, 1 - ax], u_tr[:, ax], degree=degree)
            pred = _predict_cross(q_te, own, cross, ax)
            rmsec[name] += len(q_te) * _rmse(pred, u_te[:, ax])

    for d in (rmse1, rmsec):
        for k in d:
            d[k] /= tot
    return rmse1, rmsec


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
    ap.add_argument("--couple", action="store_true",
                    help="额外拟合二维耦合逆映射 g(q_fb,q_lr)，仅当留出验证不劣于 1D 时才写入")
    ap.add_argument("--couple-degree", type=int, default=3, help="二维逆映射总阶数（i+j≤d）")
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

    # 逐会话加载，保留会话边界（2D 留出验证需要）
    sessions: list[tuple[np.ndarray, np.ndarray]] = []
    if args.converged_only:
        for csv_path in csv_paths:
            qs, us = load_converged([csv_path], args.threshold)
            if len(qs):
                sessions.append((qs, us))
        q = np.concatenate([s[0] for s in sessions])
        u = np.concatenate([s[1] for s in sessions])
        print(f"收敛段样本: {len(q)}（阈值 {args.threshold}°，{len(sessions)} 会话）")
        if len(q) < 30:
            print("收敛段样本太少，无法拟合", file=sys.stderr)
            return 1
    else:
        for csv_path in csv_paths:
            qq, uu, _ = load_session(csv_path)
            sessions.append((qq, uu))
        q = np.concatenate([s[0] for s in sessions])
        u = np.concatenate([s[1] for s in sessions])
        print(f"开环样本: {len(q)}")

    gain_poly = {}
    for ax, name in enumerate(["fb", "lr"]):
        coeffs = fit_inverse_poly(q[:, ax], u[:, ax], degree=args.degree)
        gain_poly[name] = [float(c) for c in coeffs]
        print(f"1D {name}: " + "  ".join(f"{c:+.6f}" for c in coeffs))

        # 护栏：跳过方向分段增益、直接把 g(q) 当级 1 时，必须保证 g(q) 在工作区单调。
        # 折叠（v6）会让前馈劣于纯 PID；此时应回退 direction_gains 或降阶/扩覆盖。
        q_rng = (float(q[:, ax].min()), float(q[:, ax].max()))
        mono, mind = inverse_is_monotonic(coeffs, q_rng)
        if not mono:
            print(f"  ⚠️ {name} 逆映射在 {q_rng[0]:.0f}~{q_rng[1]:.0f}° 内非单调"
                  f"（min g'(q)={mind:+.4f}）→ 会折叠，勿直接上机；"
                  f"回退 direction_gains 或降 --degree / 扩大开环激励覆盖")

    data: dict = {"gain_poly": gain_poly}

    # 几何耦合解耦交叉项（加性）：数据驱动护栏——只有留出验证不劣于 1D 才写入。
    # 写入后 ControllerConfig 在 own(q_self) 基础上叠加 cross(q_other)；不写入则退回 v7。
    if args.couple:
        own_all, cross_all = {}, {}
        for ax, name in enumerate(["fb", "lr"]):
            own, cross = fit_additive_cross(q[:, ax], q[:, 1 - ax], u[:, ax],
                                            degree=args.couple_degree)
            own_all[name] = [float(c) for c in own]
            cross_all[name] = [float(c) for c in cross]
            other = "lr" if ax == 0 else "fb"
            print(f"交叉项 {name}←{other}: " + "  ".join(f"{c:+.4f}" for c in cross))

        adopt = False
        if len(sessions) >= 2:
            r1, rc = _loso_compare(sessions, args.couple_degree)
            print(f"留出验证 RMSE(offset) 1D: fb={r1['fb']:.2f} lr={r1['lr']:.2f}  |  "
                  f"交叉: fb={rc['fb']:.2f} lr={rc['lr']:.2f}")
            adopt = rc["fb"] <= r1["fb"] and rc["lr"] <= r1["lr"]
            print(f"护栏判定: 交叉项 {'✅ 不劣于 1D，写入' if adopt else '❌ 劣于 1D，不写入（回退 1D）'}")
        else:
            # 单会话：无留出集，仅报样本内，明确警示过拟合风险
            r1 = {"fb": _rmse(_predict_1d(q[:, 0], fit_inverse_poly(q[:, 0], u[:, 0], args.degree)), u[:, 0]),
                  "lr": _rmse(_predict_1d(q[:, 1], fit_inverse_poly(q[:, 1], u[:, 1], args.degree)), u[:, 1])}
            rc = {"fb": _rmse(_predict_cross(q, np.asarray(own_all["fb"]), np.asarray(cross_all["fb"]), 0), u[:, 0]),
                  "lr": _rmse(_predict_cross(q, np.asarray(own_all["lr"]), np.asarray(cross_all["lr"]), 1), u[:, 1])}
            print(f"⚠️ 单会话，仅样本内对比（有过拟合风险，需实机 A/B 把关）")
            print(f"样本内 RMSE(offset) 1D: fb={r1['fb']:.2f} lr={r1['lr']:.2f}  |  "
                  f"交叉: fb={rc['fb']:.2f} lr={rc['lr']:.2f}")
            adopt = False

        if adopt:
            # 运行时 own 与 cross 须来自同一次联合拟合，故用联合 own 覆盖 gain_poly
            data["gain_poly"] = own_all
            data["gain_cross"] = cross_all

    if args.slew_limit is not None:
        data["slew_limit"] = args.slew_limit

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
