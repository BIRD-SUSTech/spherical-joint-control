"""训练↔运行时**特征一致性**验证 + 零回归保证验证。

两个最容易静默出错的点，必须机器验证（不能靠读代码）：
  1. 运行时 `SeqFeatureBuffer` 拼出的特征窗，必须与训练 `build_dataset` 的窗口**逐元素一致**。
     不一致 → 上机表现与离线报告无关（train/serve skew），且不会报错。
  2. `cap=0` 时必须**逐位**等于纯静态基座（零回归保证）。

用法：
    python -m model.verify_seq_runtime --config configs/dynamic_seq_v4.json \
        --session <会话目录>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.controller_config import ControllerConfig     # noqa: E402
from control.forward_seq_runtime import SeqFeatureBuffer   # noqa: E402
from model.fit_dynamic_seq import build_dataset, N_FEAT    # noqa: E402
from model.fit_dynamic_nn import SENSOR_COLS               # noqa: E402


def _rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description="时序模型 训练↔运行时一致性验证")
    ap.add_argument("--config", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--vel-w", type=int, default=31)
    args = ap.parse_args()

    d = Path(args.session)
    cfg = ControllerConfig.load(args.config)
    ref = cfg.dynamic_seq
    print(f"配置 {args.config}\n  model={ref['model']} seq_len={ref.get('seq_len')} "
          f"vel_w={ref.get('vel_w')} cap={ref.get('cap')} device={ref.get('device')}")
    print(f"  模型已载入: seq_len={cfg._seq.seq_len} 特征 {len(cfg._seq.features)} 维")

    # ---- 训练侧窗口 ----
    base = ControllerConfig.load(ref["base_config"])
    base.slew_limit = None; base.dynamic_nn = None
    W_tr, Y_tr, S, I, feats, u, qd, du, dt_tr = build_dataset(
        d, cfg._seq.seq_len, 5, args.vel_w, 1.0, base)

    # ---- 运行时侧：把同一会话的原始流逐拍喂进缓冲 ----
    srows = [r for r in _rows(d / "servo_data.csv")
             if r.get("segment_id", "") not in ("", "-1", None)]
    for r in srows:
        r["_t"] = float(r["t_s"])
    srows.sort(key=lambda r: r["_t"])
    ts = np.array([int(r["pc_receive_unix_time_ms"]) for r in srows])
    # 生产路径：dt 取配置值（训练与运行时**必须同值**；因果核按它设计）。
    # 实测采样间隔约 0.0101s，与标称 0.01 差 1% → 导数特征差 ~2%，远小于特征 std，可接受。
    dt = float(ref.get("dt", 0.01))
    print(f"  dt(配置)={dt}  训练侧实测={dt_tr:.5f}  本会话实测="
          f"{np.median(np.diff([r['_t'] for r in srows])):.5f}")

    def _sensor(fn, cols):
        rs = _rows(d / fn)
        t = np.array([int(r["pc_receive_unix_time_ms"]) for r in rs])
        o = np.argsort(t)
        return t[o], np.array([[float(r[SENSOR_COLS[c]]) for c in cols] for r in rs])[o]

    it, iv = _sensor("imu_data.csv", ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])
    ft, fv = _sensor("force_data.csv", ["ch1", "ch2", "ch3", "ch4"])
    ii = np.clip(np.searchsorted(it, ts), 0, len(it) - 1)
    fi = np.clip(np.searchsorted(ft, ts), 0, len(ft) - 1)

    q = np.array([[float(r["current_front_back_deg"]), float(r["current_left_right_deg"])]
                  for r in srows])
    qd_ = np.array([[float(r["target_front_back_deg"]), float(r["target_left_right_deg"])]
                    for r in srows])
    uu = np.array([[float(r["servo_front_back_offset"]), float(r["servo_left_right_offset"])]
                   for r in srows])
    ub = np.array([base.feedforward(qd_[i, 0], qd_[i, 1], 0.0, 0.0) for i in range(len(srows))])

    buf = SeqFeatureBuffer(cfg._seq.seq_len, args.vel_w, dt)
    win_by_t, du_rt, ubase_rt = {}, {}, {}
    for i in range(len(srows)):
        prev = (uu[i - 1, 0], uu[i - 1, 1]) if i > 0 else (uu[i, 0], uu[i, 1])
        buf.push(q[i, 0], q[i, 1], q_d_fb=qd_[i, 0], q_d_lr=qd_[i, 1], u_prev=prev,
                 gyro=iv[ii[i], :3], acc=iv[ii[i], 3:], force=fv[fi[i]])
        if buf.ready():
            win_by_t[i] = buf.window().astype(np.float64)
            p = cfg._seq.predict()
            du_rt[i] = p
            ubase_rt[i] = ub[i]

    # ---- 比一致性：对齐训练窗口的末拍索引 ----
    idx = np.array([i for i in I if i in win_by_t])
    print(f"\n训练窗口 {len(W_tr)} 个，运行时窗口 {len(win_by_t)} 个，可比对 {len(idx)} 个")
    if len(idx) == 0:
        print("❌ 无可比对窗口"); return 1
    tr_map = {int(t): k for k, t in enumerate(I)}
    diff, worst = [], None
    for t in idx[:4000]:
        a = W_tr[tr_map[int(t)]].astype(np.float64)
        b = win_by_t[int(t)]
        dd = np.abs(a - b)
        diff.append(dd.max())
        if worst is None or dd.max() > worst[1]:
            worst = (int(t), float(dd.max()), int(np.argmax(dd.max(axis=0))))
    diff = np.array(diff)
    print(f"特征逐元素最大绝对差：中位 {np.median(diff):.3e}  最大 {diff.max():.3e}"
          f"  (最差拍 {worst[0]}，特征列 #{worst[2]} = {cfg._seq.features[worst[2]]})")
    # 逐列最大差（诊断用：哪一列漂移一眼可见）
    cols = np.zeros(N_FEAT)
    for t in idx[:4000]:
        cols = np.maximum(cols, np.abs(W_tr[tr_map[int(t)]].astype(np.float64)
                                       - win_by_t[int(t)]).max(axis=0))
    # 判据用**相对容差**：每列最大差 < 该列标准差的 0.5%（绝对阈值对量纲无意义）
    std = W_tr.reshape(-1, N_FEAT).std(0) + 1e-9
    rel = cols / std
    ok_feat = rel.max() < 5e-3
    bad_cols = [f"{cfg._seq.features[k]}={rel[k]:.1e}" for k in range(N_FEAT) if rel[k] > 5e-3]
    print(f"  相对容差 max(|Δ|/std) = {rel.max():.2e}（判据 <5e-3）"
          + ("  ✅ 特征一致" if ok_feat else "  ❌ train/serve skew！"))
    print("  超差列：" + (", ".join(bad_cols) if bad_cols else "无"))
    # ---- 零回归：cap=0 必须逐位等于纯静态基座 ----
    c0 = ControllerConfig.load(args.config)
    c0.dynamic_seq = dict(c0.dynamic_seq); c0.dynamic_seq["cap"] = 0.0
    c0._init_seq(Path(args.config).parent)
    c1 = ControllerConfig.load(args.config)
    c0._seq.reset(); c1._seq.reset()
    from control.forward_seq_runtime import SeqFeatureBuffer as _B
    c0._seq.buf = _B(c0._seq.seq_len, args.vel_w, dt); c1._seq.buf = _B(c1._seq.seq_len, args.vel_w, dt)
    bad = 0; checked = 0
    for i in range(len(srows)):
        prev = (uu[i - 1, 0], uu[i - 1, 1]) if i > 0 else (uu[i, 0], uu[i, 1])
        kw = dict(q_d_fb=qd_[i, 0], q_d_lr=qd_[i, 1],
                  gyro=iv[ii[i], :3], acc=iv[ii[i], 3:], force=fv[fi[i]])
        c0.push_runtime_state(q[i, 0], q[i, 1], prev[0], prev[1], **kw)
        c1.push_runtime_state(q[i, 0], q[i, 1], prev[0], prev[1], **kw)
        a = c0.feedforward(qd_[i, 0], qd_[i, 1], 0.0, 0.0)
        b = c1.feedforward(qd_[i, 0], qd_[i, 1], 0.0, 0.0)
        if a != b:
            bad += 1
        checked += 1
    print(f"\ncap=0 零回归：{checked} 拍中 {bad} 拍与纯静态基座不同 "
          + ("✅ 逐位一致" if bad == 0 else "❌ 零回归被破坏"))
    print(f"cap>0 时序项统计：{c1.seq_stats()}")
    return 0 if (ok_feat and bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
