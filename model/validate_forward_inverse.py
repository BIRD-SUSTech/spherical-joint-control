"""§8.4 反解 F 模式：离线验证——**反解是否可信**。

反解 F 的风险不在"模型准不准"，而在**循环性**：若训练数据里 u 本身就是由控制器
按状态算出来的，模型可能学成"从历史里猜 u"而不是"从 u 算加速度"，此时
∂F/∂u 是虚假的 → 反解出的 u 无意义、上机就是随机扰动。

本工具用三组独立证据判定：

  (A) 灰箱线性回归参照：q̈ ≈ a0 + a1·q + a2·q̇ + a3·u（最小二乘）→ a3 即 ∂q̈/∂u 的
      **独立估计**（不依赖 GRU）。GRU 的 ∂F/∂u 必须与之同量级、同号。
  (B) 反解自洽：把**记录到的** q̈ 作为目标反解 → 解出的 u 应≈记录到的 u。
  (C) **反循环对照实验**：训练时把头部输入的 u 打乱（u_t 变成随机动作）再训一遍。
      若留出 RMSE 几乎不变 → 模型本来就没在用 u_t → 反解无信号（否决）；
      若显著变差 → u_t 携带真实因果信息 → 反解有信号。

用法：
    python -m model.validate_forward_inverse --session <A> <B> <C> <D> --loso
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.fit_forward_seq import (load_session, build_windows, train, export,  # noqa: E402
                                   N_FEAT)
from control.forward_seq_runtime import ForwardSeqModel, poly_pinv  # noqa: E402


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def greybox_dqddot_du(X, Yc):
    """灰箱线性回归 q̈ ~ [1, q, q̇, u] → 返回 (coef_u (2,), R²)。X 列: q(0,1) v(2,3) u(4,5)。"""
    A = np.column_stack([np.ones(len(X)), X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]])
    out = {}
    for a, ax in enumerate(["fb", "lr"]):
        y = Yc[:, a]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1.0 - np.var(y - A @ c) / max(np.var(y), 1e-9)
        out[ax] = {"coef_u": float(c[5 + a]), "r2": float(r2)}
    return out


def _poly_full(x, dt, w):
    """全序列因果多项式估计 → (m,2,3) 对齐到行号（前 w-1 拍用首值填充）。"""
    M = poly_pinv(w, dt)
    win = np.lib.stride_tricks.sliding_window_view(x, w, axis=0)
    e = np.einsum("ij,mcj->mci", M, win)
    out = np.empty((len(x), x.shape[1], 3))
    out[w - 1:] = e
    out[:w - 1] = e[0]
    return out


def diagnose(model: ForwardSeqModel, W, Yt, U, I, AUX, seq_len, vel_w, dt, tag,
             horizon: int = 0):
    """对一个留出会话跑 (A)(B)(C) 诊断。"""
    n = len(W)
    h_cache = [model.encode(W[i]) for i in range(n)]

    # --- 预测精度 ---
    pred = np.array([model.head(h_cache[i], U[i]) for i in range(n)])
    r = {"n": n, "tag": tag}
    for a, ax in enumerate(["fb", "lr"]):
        r[f"rmse_{ax}"] = _rmse(pred[:, a], Yt[:, a])
        r[f"base_{ax}"] = _rmse(np.zeros_like(Yt[:, a]), Yt[:, a])

    # --- (A) 灰箱参照 vs GRU 灵敏度 ---
    Xh = np.array([W[i, -1] for i in range(n)])
    Yc = AUX["qddot_causal"][I] if horizon == 0 else Yt
    r["greybox"] = greybox_dqddot_du(Xh, Yc)
    sub = np.arange(0, n, max(1, n // 300))
    J = np.array([model.jac_u(h_cache[i], U[i]) for i in sub])
    r["jac"] = {"diag_fb": float(np.mean(J[:, 0, 0])), "diag_lr": float(np.mean(J[:, 1, 1])),
                "off_01": float(np.mean(J[:, 0, 1])), "off_10": float(np.mean(J[:, 1, 0])),
                "det_abs_mean": float(np.mean(np.abs(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]))),
                "diag_fb_std": float(np.std(J[:, 0, 0])), "diag_lr_std": float(np.std(J[:, 1, 1]))}

    # --- (B) 反解自洽：目标 = 记录到的 q̈ → 解出的 u 应≈记录到的 u ---
    Urec = AUX["u"][I]
    i2 = np.arange(0, n, max(1, n // 200))
    u_inv, res = [], []
    for i in i2:
        uu, info = model.solve_u(h_cache[i], Yt[i], U[i], iters=10)
        u_inv.append(uu); res.append(info["resid"])
    u_inv = np.array(u_inv)
    r["selfconsist"] = {"du_rmse_fb": _rmse(u_inv[:, 0], Urec[i2, 0]),
                        "du_rmse_lr": _rmse(u_inv[:, 1], Urec[i2, 1]),
                        "u_std_fb": float(np.std(Urec[i2, 0])), "u_std_lr": float(np.std(Urec[i2, 1])),
                        "resid_med": float(np.median(res))}

    # --- (C) 反事实：目标 = 解析期望轨迹的 q̈_d ---
    # 全序列上算因果多项式估计，再按行号 I 取值（I 是稀疏行号，不能先取子集再滑窗）
    pd = _poly_full(AUX["q_d"], dt, vel_w)             # 期望轨迹的 (q, q̇, q̈)
    if horizon > 0:
        # 视界模式目标：q̇_d(t+H) − q̇_meas(t)（期望轨迹的解析速度 − 当前实测速度）
        pm = _poly_full(AUX["q"], dt, vel_w)
        idx_h = np.clip(I + horizon, 0, len(AUX["q_d"]) - 1)
        qd_a = pd[idx_h, :, 1] - pm[I, :, 1]
    else:
        qd_a = pd[I, :, 2]                             # (n,2) 目标加速度
    u_cf, nfail = [], 0
    for i in i2:
        uu, info = model.solve_u(h_cache[i], qd_a[i], U[i], iters=12)
        if info["resid"] > 1.0:
            nfail += 1
        u_cf.append(uu)
    u_cf = np.array(u_cf)
    r["counterfactual"] = {
        "du_med_fb": float(np.median(np.abs(u_cf[:, 0] - Urec[i2, 0]))),
        "du_med_lr": float(np.median(np.abs(u_cf[:, 1] - Urec[i2, 1]))),
        "du_p95_fb": float(np.percentile(np.abs(u_cf[:, 0] - Urec[i2, 0]), 95)),
        "du_p95_lr": float(np.percentile(np.abs(u_cf[:, 1] - Urec[i2, 1]), 95)),
        "u_inv_absmax_fb": float(np.max(np.abs(u_cf[:, 0]))),
        "u_inv_absmax_lr": float(np.max(np.abs(u_cf[:, 1]))),
        "unsolved_frac": nfail / max(len(i2), 1),
    }
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="反解 F 模式离线验证")
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--smooth", type=int, default=25)
    ap.add_argument("--acc-h", type=int, default=15)
    ap.add_argument("--vel-w", type=int, default=31)
    ap.add_argument("--horizon", type=int, default=0, help="跨拍视界 H（拍）；0=单拍 q̈")
    ap.add_argument("--no-u-hist", action="store_true",
                    help="历史特征不含 u（与训练一致；反解必需）")
    ap.add_argument("--loso", action="store_true", help="逐会话留出（否则单次 train-all）")
    ap.add_argument("--ctrl-shuffle-u", action="store_true", help="跑反循环对照（打乱 u 再训）")
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    args = ap.parse_args()

    parts = []
    for s in args.session:
        p = Path(s)
        d = p if p.is_dir() else p.parent
        r = load_session(d, args.smooth, args.acc_h, "poly", args.vel_w, with_aux=True,
                         use_u_hist=not args.no_u_hist, horizon=args.horizon)
        if r is None:
            print(f"跳过 {d.name}", file=sys.stderr)
            continue
        w = build_windows(r[0], r[1], r[2], args.seq_len, args.stride,
                          y_shift=0 if args.horizon > 0 else 1)
        dt = float(np.median(np.diff(r[3]["t"])))
        parts.append({"name": d.name, "X": r[0], "Y": r[1], "S": r[2], "AUX": r[3],
                      "win": w, "dt": dt})
        print(f"{d.name}: {len(r[0])} 拍 / {len(w[0])} 窗 / dt={dt*1000:.1f}ms")

    results = []
    if args.loso:
        for h in range(len(parts)):
            tr = [p for i, p in enumerate(parts) if i != h]
            te = parts[h]
            W = np.concatenate([p["win"][0] for p in tr])
            Yt = np.concatenate([p["win"][1] for p in tr])
            U = np.concatenate([p["win"][2] for p in tr])
            print(f"\n=== 留出 {te['name']}（训练 {len(W)} 窗） ===")
            net, norm, _ = train(W, Yt, U, args.hidden, args.epochs)
            tmp = Path("/tmp/_fs_loso.npz")
            d0 = export(net, norm, args.hidden, u_hist=not args.no_u_hist)
            np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in d0.items()})
            model = ForwardSeqModel(tmp)
            Wh, Yh, Uh, Sh, Ih = te["win"]
            r = diagnose(model, Wh, Yh, Uh, Ih, te["AUX"], args.seq_len, args.vel_w,
                         te["dt"], te["name"], args.horizon)
            _print(r)
            results.append(r)
    else:
        W = np.concatenate([p["win"][0] for p in parts])
        Yt = np.concatenate([p["win"][1] for p in parts])
        U = np.concatenate([p["win"][2] for p in parts])
        net, norm, _ = train(W, Yt, U, args.hidden, args.epochs)
        tmp = Path("/tmp/_fs_all.npz")
        d0 = export(net, norm, args.hidden, u_hist=not args.no_u_hist)
        np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in d0.items()})
        model = ForwardSeqModel(tmp)
        for p in parts:
            Wh, Yh, Uh, Sh, Ih = p["win"]
            r = diagnose(model, Wh, Yh, Uh, Ih, p["AUX"], args.seq_len, args.vel_w,
                         p["dt"], p["name"], args.horizon)
            _print(r)
            results.append(r)

    # --- 反循环对照 ---
    if args.ctrl_shuffle_u:
        print("\n" + "=" * 68)
        print("反循环对照：训练时打乱头部输入 u（u_t → 随机动作）")
        print("  判据：若留出 RMSE 与真实模型【几乎相同】→ 模型没用 u_t → 反解无信号")
        W = np.concatenate([p["win"][0] for p in parts])
        Yt = np.concatenate([p["win"][1] for p in parts])
        U = np.concatenate([p["win"][2] for p in parts])
        rng = np.random.default_rng(0)
        Us = U[rng.permutation(len(U))]
        net2, norm2, _ = train(W, Yt, Us, args.hidden, args.epochs)
        import torch
        with torch.no_grad():
            wt = torch.tensor((W - norm2["xm"]) / norm2["xs"], dtype=torch.float32)
            ptr = net2(wt, torch.tensor((U - norm2["um"]) / norm2["us"], dtype=torch.float32)).numpy() * norm2["ys"] + norm2["ym"]
            ptc = net2(wt, torch.tensor((Us - norm2["um"]) / norm2["us"], dtype=torch.float32)).numpy() * norm2["ys"] + norm2["ym"]
        # 对照2（最干净）：**完全去掉 u 输入**（训练+推理都把 u 列置零）
        U0 = np.zeros_like(U)
        net3, norm3, _ = train(W, Yt, U0, args.hidden, args.epochs)
        with torch.no_grad():
            wt3 = torch.tensor((W - norm3["xm"]) / norm3["xs"], dtype=torch.float32)
            p0 = net3(wt3, torch.tensor((U0 - norm3["um"]) / norm3["us"], dtype=torch.float32)).numpy() * norm3["ys"] + norm3["ym"]
        for a, ax in enumerate(["fb", "lr"]):
            b = _rmse(np.zeros_like(Yt[:, a]), Yt[:, a])
            rr = _rmse(ptr[:, a], Yt[:, a])
            rc = _rmse(ptc[:, a], Yt[:, a])
            r0 = _rmse(p0[:, a], Yt[:, a])
            print(f"  [{ax}] 基线 {b:6.2f} | 含u {rr:6.2f} | 打乱u {rc:6.2f} ({(rc/rr-1)*100:+.1f}%) "
                  f"| **无u** {r0:6.2f} ({(r0/rr-1)*100:+.1f}%)")
            results.append({"ctrl": ax, "base": b, "with_u": rr, "shuf_u": rc, "no_u": r0})

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\n已写 {args.out}")
    return 0


def _print(r):
    g, j, s, c = r["greybox"], r["jac"], r["selfconsist"], r["counterfactual"]
    print(f"  样本内/留出 RMSE  fb {r['rmse_fb']:.2f} (基线 {r['base_fb']:.2f}) | "
          f"lr {r['rmse_lr']:.2f} (基线 {r['base_lr']:.2f})")
    print(f"  (A) ∂q̈/∂u 灰箱参照 fb {g['fb']['coef_u']:+.2f} (R²={g['fb']['r2']:.3f}) | "
          f"lr {g['lr']['coef_u']:+.2f} (R²={g['lr']['r2']:.3f})")
    print(f"      GRU 灵敏度  diag fb {j['diag_fb']:+.2f}±{j['diag_fb_std']:.2f} | "
          f"lr {j['diag_lr']:+.2f}±{j['diag_lr_std']:.2f} | 交叉 {j['off_01']:+.2f}/{j['off_10']:+.2f} | "
          f"|det| {j['det_abs_mean']:.1f}")
    print(f"  (B) 反解自洽（目标=记录q̈） Δu RMSE fb {s['du_rmse_fb']:.2f} "
          f"(u 波动 {s['u_std_fb']:.1f}) | lr {s['du_rmse_lr']:.2f} (u 波动 {s['u_std_lr']:.1f}) "
          f"| 残差中位 {s['resid_med']:.3f}")
    print(f"  (C) 反事实（目标=q̈_d） |Δu| 中位 fb {c['du_med_fb']:.2f} / lr {c['du_med_lr']:.2f} | "
          f"p95 {c['du_p95_fb']:.1f} / {c['du_p95_lr']:.1f} | "
          f"|u|max {c['u_inv_absmax_fb']:.0f}/{c['u_inv_absmax_lr']:.0f} | "
          f"未解出 {c['unsolved_frac']*100:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
