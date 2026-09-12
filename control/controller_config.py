"""控制器参数配置（独立于标定，为后续前馈项留接口）。

职责拆分：
    - 标定 Calibration = 认识系统的结果：符号映射 + 基础增益（随硬件平台，相对固定）。
    - 控制器参数 ControllerConfig = 控制策略：前馈增益调度 + 各种前馈项（随优化迭代更新）。

前馈项接口（可扩展，新增前馈项时在此追加字段并在 load 里读取）：
    - direction_gains：方向分段增益（级 1，已实现）
    - gain_poly：参数化逆映射 g(q) 系数（级 1.5，已实现）
    - gain_cross：几何耦合解耦交叉项 c(q_other)（级 1.6，已实现，加性）
    - velocity_gain：速度前馈（加性线性）u_ff = g(q_d) + c·q̇_d（§8.4，D0 实证最佳形式）
    - velocity_lead：速度前馈（相位超前式）u_ff = g(q_d + τ·q̇_d)（等价于系数 g′(q)·τ，
      带 q 依赖；实测离线 RMSE 比加性式差 48%，保留供 A/B 对照）
    - dynamic_nn：学习型动态残差 MLP（§8.4，已实现）——**残差式**，权重置零=精确退回基座。
      权重可**内联在 JSON**（dynamic_nn）或**存 npz**（dynamic_nn_npz，推荐：JSON 只留引用、
      权重用二进制，避免 47KB JSON 嵌套列表）。
    - hysteresis：迟滞补偿（级 2，预留）
    - friction：摩擦补偿（级 3，预留）
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ControllerConfig:
    direction_gains: dict | None = None   # {"fb": {"pos","neg"}, "lr": {"pos","neg"}}
    gain_poly: dict | None = None         # {"fb": [b0,b1,b2,b3], "lr": [...]} 逆映射 g(q_self)
    gain_cross: dict | None = None        # {"fb": [c1..], "lr": [c1..]} 交叉项 c(q_other)，无常数项
    velocity_gain: dict | None = None     # {"fb": c, "lr": c} 速度前馈加性系数 offset/(°/s)
    velocity_lead: dict | None = None     # {"fb": tau_s, "lr": tau_s} 速度前馈相位超前（秒）
    dynamic_nn: dict | None = None        # 动态残差 MLP（内联形式）：归一化参数 + 层权重
    dynamic_nn_npz: str | None = None     # 动态残差 MLP 权重文件（npz）；load 时载入 dynamic_nn
    dynamic_seq: dict | None = None       # 反解 F 模式（GRU 时序模型 + 运行时反解）；见 _init_seq
    slew_limit: float | None = None       # u_ff 每拍变化上限（offset/拍），None=不限
    hysteresis: dict | None = None        # {"fb": h, "lr": h} 迟滞补偿（offset），级 2
    # 未来扩展字段在此追加，load 时读取对应 key

    _last_uff: tuple = field(default=None, init=False, repr=False)  # 上次前馈输出（slew 用）
    _seq: object = field(default=None, init=False, repr=False)      # SeqRuntime（torch）
    _seq_stat: dict = field(default=None, init=False, repr=False)   # 时序项运行统计（诊断用）

    def __post_init__(self):
        self._last_uff = None
        self._seq_stat = {"n": 0, "du_fb": [], "du_lr": [], "skip": 0}

    @classmethod
    def none(cls) -> "ControllerConfig":
        """无前馈（u_ff=0）。"""
        return cls(direction_gains=None)

    @classmethod
    def load(cls, path: str | Path) -> "ControllerConfig":
        cfg_path = Path(path)
        data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        cfg = cls(
            direction_gains=data.get("direction_gains"),
            gain_poly=data.get("gain_poly"),
            gain_cross=data.get("gain_cross"),
            velocity_gain=data.get("velocity_gain"),
            velocity_lead=data.get("velocity_lead"),
            dynamic_nn=data.get("dynamic_nn"),
            dynamic_nn_npz=data.get("dynamic_nn_npz"),
            dynamic_seq=data.get("dynamic_seq"),
            slew_limit=data.get("slew_limit"),
            hysteresis=data.get("hysteresis"),
        )
        # npz 权重（推荐形式）：JSON 只留引用，载入时materialize成 dynamic_nn
        if cfg.dynamic_nn is None and cfg.dynamic_nn_npz:
            ref = Path(cfg.dynamic_nn_npz)
            if not ref.is_absolute():
                ref = cfg_path.parent / ref
            cfg.dynamic_nn = load_nn_npz(ref)
        if cfg.dynamic_seq:
            cfg._init_seq(cfg_path.parent)
        return cfg

    # ------------------------------------------------------------------
    # §8.4 反解 F 模式：GRU 时序模型 + 运行时反解（纯模型动态前馈）
    # ------------------------------------------------------------------
    def _init_seq(self, base_dir: Path) -> None:
        """载入 torch 时序模型 + 建因果特征缓冲。配置见 configs/*.json 的 dynamic_seq 段。

        dynamic_seq 字段：
            model    模型文件（.pt，torch.save）路径，相对配置文件目录
            seq_len  历史窗长（拍），须与训练一致（缺省 10）
            vel_w    q̇ 因果估计器窗长（拍），须与训练一致（缺省 31）
            dt       控制周期秒（缺省 0.01）；仅用于 q̇ 核
            device   "cpu"（缺省，控制回路不需要 GPU）| "auto" | "cuda"
            cap      修正量幅值帽（offset）；**0 = 逐位退回静态前馈**（缺省 0，安全）
        """
        from control.forward_seq_runtime import SeqRuntime
        ref = Path(self.dynamic_seq["model"])
        if not ref.is_absolute():
            ref = base_dir / ref
        self._seq = SeqRuntime(ref,
                               seq_len=int(self.dynamic_seq.get("seq_len", 10)),
                               vel_w=int(self.dynamic_seq.get("vel_w", 31)),
                               dt=float(self.dynamic_seq.get("dt", 0.01)),
                               device=str(self.dynamic_seq.get("device", "cpu")))
        self._seq_ref = str(ref)

    def push_runtime_state(self, q_meas_fb, q_meas_lr, u_prev_fb, u_prev_lr,
                           q_d_fb=0.0, q_d_lr=0.0,
                           gyro=None, acc=None, force=None) -> bool:
        """每拍喂入【当前测量 + 期望轨迹 + 上一拍实发动作】→ 因果特征窗。

        必须在 feedforward() **之前**调用（仅 dynamic_seq 启用时有效）。
        u_prev 必须是**实际下发**的总 offset（u_ff+u_fb），与训练标签语义一致。
        返回 False = 本拍不可用（缓冲未满/未启用）→ 自动退回静态前馈。
        """
        if self._seq is None:
            return False
        self._seq.push(q_meas_fb, q_meas_lr, q_d_fb=q_d_fb, q_d_lr=q_d_lr,
                       u_prev=(u_prev_fb, u_prev_lr),
                       gyro=(0.0, 0.0, 0.0) if gyro is None else gyro,
                       acc=(0.0, 0.0, 0.0) if acc is None else acc,
                       force=(0.0, 0.0, 0.0, 0.0) if force is None else force)
        return self._seq.ready()

    def seq_stats(self) -> dict:
        """时序项运行统计（诊断：修正量分布 / 未生效拍数）。"""
        st = dict(self._seq_stat)
        for k in ("du_fb", "du_lr"):
            v = st.pop(k, [])
            if v:
                import statistics as _s
                st[k + "_med"] = _s.median(v)
                st[k + "_p95"] = sorted(v)[int(0.95 * (len(v) - 1))]
                st[k + "_max"] = max(v)
                st[k + "_absmax"] = max(abs(x) for x in v)
        return st

    def has_feedforward(self) -> bool:
        """是否启用前馈。"""
        return (self.direction_gains is not None or self.gain_poly is not None
                or self.gain_cross is not None or self.dynamic_nn is not None
                or self.velocity_gain is not None or self.velocity_lead is not None
                or self.dynamic_seq is not None)

    def feedforward(self, q_d_fb: float, q_d_lr: float,
                    qdot_d_fb: float = 0.0, qdot_d_lr: float = 0.0,
                    qddot_d_fb: float = 0.0, qddot_d_lr: float = 0.0,
                    gyro=None, acc=None, force=None) -> tuple[float, float]:
        """前馈反解：目标关节角（度）→ 差分 offset（含迟滞 + slew 整形）。

        静态基座 = own(q_self) + 交叉项 c(q_other)；own 依次回退
        gain_poly > direction_gains；级 2 = 迟滞项 h·sign(q̇_d)。
        """
        if self.gain_poly is not None:
            # 速度前馈（相位超前）：把静态逆映射按 τ·q̇_d 前瞻求值。
            # τ=0（或缺省）→ 与 g(q_d) **逐位相同**，零退回保证天然成立。
            q_eval_fb, q_eval_lr = q_d_fb, q_d_lr
            if self.velocity_lead is not None:
                q_eval_fb += self.velocity_lead.get("fb", 0.0) * qdot_d_fb
                q_eval_lr += self.velocity_lead.get("lr", 0.0) * qdot_d_lr
            u_fb = _poly(self.gain_poly["fb"], q_eval_fb)
            u_lr = _poly(self.gain_poly["lr"], q_eval_lr)
        elif self.direction_gains is not None:
            g = self.direction_gains
            u_fb = q_d_fb / (g["fb"]["pos"] if q_d_fb >= 0 else g["fb"]["neg"])
            u_lr = q_d_lr / (g["lr"]["pos"] if q_d_lr >= 0 else g["lr"]["neg"])
        else:
            u_fb = u_lr = 0.0

        # 速度前馈（加性线性）：u_ff += c·q̇_d（D0 干净数据实测最优形式）
        if self.velocity_gain is not None:
            u_fb += self.velocity_gain.get("fb", 0.0) * qdot_d_fb
            u_lr += self.velocity_gain.get("lr", 0.0) * qdot_d_lr

        # 级 1.6：几何耦合解耦交叉项（加性，无常数项）
        if self.gain_cross is not None:
            u_fb += _poly_no_const(self.gain_cross["fb"], q_d_lr)  # fb 输出受 lr 角影响
            u_lr += _poly_no_const(self.gain_cross["lr"], q_d_fb)  # lr 输出受 fb 角影响

        # §8.4 D1：学习型动态残差（**残差式**，只加修正量；缺省/零权重=退回稳态前馈）
        if self.dynamic_nn is not None:
            src = {"q_d_fb": q_d_fb, "q_d_lr": q_d_lr,
                   "qdot_d_fb": qdot_d_fb, "qdot_d_lr": qdot_d_lr,
                   "qddot_d_fb": qddot_d_fb, "qddot_d_lr": qddot_d_lr}
            if gyro is not None:
                src.update(zip(("gyro_x", "gyro_y", "gyro_z"), gyro))
            if acc is not None:
                src.update(zip(("acc_x", "acc_y", "acc_z"), acc))
            if force is not None:
                src.update(zip(("ch1", "ch2", "ch3", "ch4"), force))
            names = self.dynamic_nn.get("features")
            if names:
                # 按**特征名**取值（单一事实源）→ 支持 v4(含 IMU/力) 且不怕顺序变化
                feats = [float(src.get(str(nm), 0.0)) for nm in names]
            else:                                          # 旧模型无 features 字段：位置式
                feats = [q_d_fb, q_d_lr, qdot_d_fb, qdot_d_lr]
                if int(self.dynamic_nn.get("n_in", 4)) >= 6:
                    feats += [qddot_d_fb, qddot_d_lr]
            r_fb, r_lr = _nn_forward(self.dynamic_nn, feats)
            u_fb += r_fb
            u_lr += r_lr

        # §8.4 时序动态前馈（论文主模型）：u_ff = u_base + clip(Δu_GRU, ±cap)
        if self._seq is not None:
            cap = float(self.dynamic_seq.get("cap", 0.0))
            du = self._seq.predict() if cap > 0.0 else None
            if du is not None:
                u_fb += _clip(du[0], cap)
                u_lr += _clip(du[1], cap)
                self._seq_stat["n"] += 1
                self._seq_stat["du_fb"].append(du[0])
                self._seq_stat["du_lr"].append(du[1])
            else:
                self._seq_stat["skip"] += 1

        # 级 2：迟滞补偿 h·sign(q̇_d)
        if self.hysteresis is not None:
            u_fb += self.hysteresis.get("fb", 0.0) * (1.0 if qdot_d_fb >= 0 else -1.0)
            u_lr += self.hysteresis.get("lr", 0.0) * (1.0 if qdot_d_lr >= 0 else -1.0)

        # slew 速率整形：限制 u_ff 每拍变化量，防目标突变时前馈跳变
        if self.slew_limit is not None:
            if self._last_uff is not None:
                u_fb = _slew(self._last_uff[0], u_fb, self.slew_limit)
                u_lr = _slew(self._last_uff[1], u_lr, self.slew_limit)
        self._last_uff = (u_fb, u_lr)
        return u_fb, u_lr


def _clip(v: float, cap: float) -> float:
    """对称幅值帽：cap<=0 → 恒 0（反解退回静态前馈的逐位保证）。"""
    if cap <= 0.0:
        return 0.0
    return max(-cap, min(cap, v))


def _poly(coeffs, x):
    """多项式求值 u = b0 + b1·x + b2·x² + ..."""
    return sum(c * x ** k for k, c in enumerate(coeffs))


def _poly_no_const(coeffs, x) -> float:
    """无常数项多项式求值 u = c1·x + c2·x² + ...（coeffs[0] 对应 x¹）。"""
    return sum(c * x ** (k + 1) for k, c in enumerate(coeffs))


def _nn_forward(nn: dict, features: list[float]) -> tuple[float, float]:
    """动态残差 MLP 前向（纯 Python，运行时无 torch 依赖）。

    nn 结构：{"in_mean","in_std","out_mean","out_std","layers":[{"W","b"}...],"act","clip"}
    输入先标准化 → 逐层 (act 除末层) → 输出去标准化 → 按 clip 限幅（防外推发散）。
    """
    x = [(f - m) / (s if s else 1.0) for f, m, s in zip(features, nn["in_mean"], nn["in_std"])]
    layers = nn["layers"]
    act = nn.get("act", "tanh")
    for i, layer in enumerate(layers):
        W, b = layer["W"], layer["b"]
        x = [sum(w * xi for w, xi in zip(row, x)) + bi for row, bi in zip(W, b)]
        if i < len(layers) - 1:
            if act == "tanh":
                x = [math.tanh(v) for v in x]
            elif act == "relu":
                x = [v if v > 0.0 else 0.0 for v in x]
    out = [v * s + m for v, m, s in zip(x, nn["out_mean"], nn["out_std"])]
    clip = nn.get("clip")
    if clip is not None:
        out = [max(-clip, min(clip, v)) for v in out]
    return out[0], out[1]


def _slew(prev: float, cur: float, limit: float) -> float:
    """斜率限制：每拍变化量不超过 limit。"""
    d = cur - prev
    if abs(d) > limit:
        return prev + limit * (1 if d > 0 else -1)
    return cur


def load_nn_npz(path: str | Path) -> dict:
    """从 npz 载入动态残差 MLP 权重 → 与内联 dynamic_nn 同构的 dict。

    npz 键：in_mean, in_std, out_mean, out_std, clip, n_in, features(可选), W0,b0,W1,b1,...
    """
    import numpy as np
    z = np.load(path, allow_pickle=False)
    layers = []
    i = 0
    while f"W{i}" in z.files:
        layers.append({"W": z[f"W{i}"].tolist(), "b": z[f"b{i}"].tolist()})
        i += 1
    out = {
        "in_mean": z["in_mean"].tolist(), "in_std": z["in_std"].tolist(),
        "out_mean": z["out_mean"].tolist(), "out_std": z["out_std"].tolist(),
        "layers": layers, "act": str(z["act"]) if "act" in z.files else "tanh",
        "clip": float(z["clip"]), "n_in": int(z["n_in"]),
        "_npz": str(path),
    }
    if "features" in z.files:
        out["features"] = [str(x) for x in z["features"]]
    return out


def save_nn_npz(path: str | Path, nn: dict) -> None:
    """把 dynamic_nn dict 写入 npz（权重二进制，JSON 不再内联大数组）。"""
    import numpy as np
    arrs = {
        "in_mean": np.asarray(nn["in_mean"], dtype=np.float64),
        "in_std": np.asarray(nn["in_std"], dtype=np.float64),
        "out_mean": np.asarray(nn["out_mean"], dtype=np.float64),
        "out_std": np.asarray(nn["out_std"], dtype=np.float64),
        "clip": np.asarray(nn.get("clip", 0.0), dtype=np.float64),
        "n_in": np.asarray(nn.get("n_in", len(nn["in_mean"])), dtype=np.int64),
        "act": np.asarray(nn.get("act", "tanh")),
    }
    if nn.get("features"):
        arrs["features"] = np.asarray(nn["features"])
    for i, L in enumerate(nn["layers"]):
        arrs[f"W{i}"] = np.asarray(L["W"], dtype=np.float64)
        arrs[f"b{i}"] = np.asarray(L["b"], dtype=np.float64)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrs)
