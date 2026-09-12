"""§8.4 A/B：给动态残差 MLP 加 IMU/力传感器输入，**到底有没有用**。

判据（吸取 q̈ 那次教训）：单看"完整模型变好了"不算证据——必须与
**「同维数但无信息」的对照组**比（打乱传感器列），且逐会话 LOSO 留出，
看 Δ 是否在会话间一致。离线只做筛选，最终仍以实机 A/B 为准。

用法：
    python -m model.ab_sensor_features --session <A> <B> <C> <D> --base-config configs/dynamic_nn_v3_static.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.controller_config import ControllerConfig  # noqa: E402
from model.fit_dynamic_nn import (load_session_features, train_mlp,  # noqa: E402
                                  feature_names)


def evaluate(per_session, hidden, depth, epochs, seed):
    """逐会话 LOSO：返回每折每轴留出 RMSE（+ 零修正基线）。"""
    out = []
    for h in range(len(per_session)):
        X_tr = np.concatenate([s[0] for i, s in enumerate(per_session) if i != h])
        R_tr = np.concatenate([s[1] for i, s in enumerate(per_session) if i != h])
        X_va, R_va = per_session[h][0], per_session[h][1]
        net, norm, _ = train_mlp(X_tr, R_tr, X_va, R_va, hidden, depth, epochs, 1e-3, seed)
        import torch
        with torch.no_grad():
            im, ist, om, ost = norm
            pred = net(torch.tensor((X_va - im) / ist, dtype=torch.float32)).numpy() * ost + om
        base = [float(np.sqrt(np.mean(R_va[:, a] ** 2))) for a in range(2)]
        mdl = [float(np.sqrt(np.mean((R_va[:, a] - pred[:, a]) ** 2))) for a in range(2)]
        out.append((base, mdl))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="IMU/力 特征 A/B（含无用特征对照）")
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--base-config", default="configs/static_feedforward_controller.json")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = ControllerConfig.load(args.base_config)
    cfg.dynamic_nn = None
    dirs = [Path(s) if Path(s).is_dir() else Path(s).parent for s in args.session]

    variants = [("v3_static(基线,无传感器)", dict(use_imu=False, use_force=False)),
                ("+IMU", dict(use_imu=True, use_force=False)),
                ("+力", dict(use_imu=False, use_force=True)),
                ("+IMU+力", dict(use_imu=True, use_force=True))]
    loaded = {}
    for name, kw in variants:
        ps = []
        for d in dirs:
            X, R, _ = load_session_features(d / "servo_data.csv", cfg, args.threshold,
                                            use_qddot=True, **kw)
            if len(X):
                ps.append((X, R))
        loaded[name] = ps
        print(f"{name}: {sum(len(p[0]) for p in ps)} 收敛样本 / 特征 "
              f"{', '.join(feature_names(True, kw['use_imu'], kw['use_force']))}")

    # 对照组：IMU/力列打乱（同维数、无信息）
    ctrl = []
    rng = np.random.default_rng(0)
    for X, R in loaded["+IMU+力"]:
        Xc = X.copy()
        Xc[:, 6:] = Xc[rng.permutation(len(Xc)), 6:]
        ctrl.append((Xc, R))
    loaded["【对照】打乱传感器列"] = ctrl

    print(f"\n{'变体':<26} {'留出残差RMSE(fb/lr，4折)':<34} {'相对基线':>10}")
    print("-" * 78)
    ref = None
    for name in list(loaded):
        res = evaluate(loaded[name], args.hidden, args.depth, args.epochs, args.seed)
        fb = np.mean([m[0] for _, m in res]); lr = np.mean([m[1] for _, m in res])
        bfb = np.mean([b[0] for b, _ in res]); blr = np.mean([b[1] for b, _ in res])
        line = f"{name:<26} fb {fb:6.2f} (基线 {bfb:6.2f})  lr {lr:6.2f} (基线 {blr:6.2f})"
        if ref is None:
            ref = (fb, lr); extra = "—"
        else:
            extra = f"fb {(fb/ref[0]-1)*100:+.1f}% lr {(lr/ref[1]-1)*100:+.1f}%"
        print(f"{line:<50} {extra:>10}")
        # 逐折一致性
        print(f"{'':<26} 逐折 fb: " + " ".join(f"{m[0]:.1f}" for _, m in res)
              + " | lr: " + " ".join(f"{m[1]:.1f}" for _, m in res))
    print("\n判据：只有「加传感器」显著优于【同维数打乱对照】+ 逐折一致，才算真有效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
