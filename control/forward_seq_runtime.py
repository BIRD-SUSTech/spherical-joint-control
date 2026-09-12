"""时序动态前馈的**运行时**：torch 模型 + 因果特征缓冲。

不手写任何前向：模型是 `model/seq_model.py` 的 `SeqFeedforward`（与训练**同一份代码**），
这里只负责

    1. `SeqFeatureBuffer` —— 把逐拍原始测量/指令拼成 22 维特征窗（**只用过去**）
    2. `SeqRuntime`       —— 载模型 + torch.no_grad 前向 → Δu

因果性：q̇ / q̈_d 用**零滞后因果多项式核**（对过去 w 拍做二次拟合、取 τ=0 处系数），
与训练完全一致；不需要未来数据、无相位滞后。

实时性：L=10、hidden=32 的 GRU 单拍前向 << 1ms（cpu），100Hz 回路绰绰有余。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model.seq_model import load_model

FEATURES = ["q_fb", "q_lr", "qdot_fb", "qdot_lr",
            "qd_fb", "qd_lr", "qdotd_fb", "qdotd_lr", "qddotd_fb", "qddotd_lr",
            "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
            "ch1", "ch2", "ch3", "ch4", "u_prev_fb", "u_prev_lr"]
N_FEAT = len(FEATURES)


def poly_kernel(w: int, dt: float) -> np.ndarray:
    """因果零滞后最小二乘核 (3,w)：行 = [q, q̇, q̈] 在当前拍（τ=0）处的估计。

    对【过去 w 拍】拟合 q(τ)=a+bτ+cτ²/2（τ≤0 为过去），取 τ=0 的系数。
    因果、零相位滞后（中心差分需要未来数据，运行时不可用）。
    """
    j = np.arange(w)
    tau = (j - (w - 1)) * dt
    a = np.stack([np.ones(w), tau, tau ** 2 / 2], axis=1)
    return np.linalg.pinv(a)


class SeqFeatureBuffer:
    """逐拍原始量 → 22 维特征窗。**只用过去**。"""

    def __init__(self, seq_len: int = 10, vel_w: int = 31, dt: float = 0.01):
        self.L = int(seq_len)
        self.w = int(vel_w)
        self.dt = float(dt)
        self.M = poly_kernel(self.w, self.dt)
        self._q: list = []          # 实测角历史（保留 w 拍，供因果核）
        self._qd: list = []         # 期望角历史
        self._rows: list = []

    def reset(self) -> None:
        self._q.clear(); self._qd.clear(); self._rows.clear()

    def _deriv(self, hist: list):
        a = np.array([p[0] for p in hist]); b = np.array([p[1] for p in hist])
        return (float(self.M[1] @ a), float(self.M[1] @ b),
                float(self.M[2] @ a), float(self.M[2] @ b))

    def push(self, q_fb, q_lr, q_d_fb=0.0, q_d_lr=0.0, u_prev=(0.0, 0.0),
             gyro=(0.0, 0.0, 0.0), acc=(0.0, 0.0, 0.0), force=(0.0, 0.0, 0.0, 0.0)) -> None:
        """喂入**当前拍**测量与**上一拍实发动作**（u_prev 须是实际下发的总 offset）。"""
        self._q.append((float(q_fb), float(q_lr)))
        self._qd.append((float(q_d_fb), float(q_d_lr)))
        if len(self._q) > self.w:
            del self._q[0]
        if len(self._qd) > self.w:
            del self._qd[0]
        if len(self._q) >= self.w:
            vf, vl, af, al = self._deriv(self._q)
            vdf, vdl, adf, adl = self._deriv(self._qd)
        else:
            # 启动瞬态：不足 w 拍 → 后向差分占位（与训练填充段一致）
            vf = vl = af = al = vdf = vdl = adf = adl = 0.0
            if len(self._q) >= 2:
                vf = (self._q[-1][0] - self._q[-2][0]) / self.dt
                vl = (self._q[-1][1] - self._q[-2][1]) / self.dt
                vdf = (self._qd[-1][0] - self._qd[-2][0]) / self.dt
                vdl = (self._qd[-1][1] - self._qd[-2][1]) / self.dt
        row = [float(q_fb), float(q_lr), vf, vl,
               float(q_d_fb), float(q_d_lr), vdf, vdl, adf, adl]
        row += [float(v) for v in gyro]
        row += [float(v) for v in acc]
        row += [float(v) for v in force]
        row += [float(u_prev[0]), float(u_prev[1])]
        self._rows.append(row)
        if len(self._rows) > self.L:
            del self._rows[0]

    def ready(self) -> bool:
        return len(self._rows) >= self.L

    def window(self):
        return np.asarray(self._rows, dtype=np.float32) if self.ready() else None


class SeqRuntime:
    """载入 torch 时序模型 + 特征缓冲 → 每拍输出 Δu。"""

    def __init__(self, model_path, seq_len: int = 10, vel_w: int = 31, dt: float = 0.01,
                 device: str = "cpu", features=None):
        self.model = load_model(Path(model_path), device=device)
        self.features = list(features or getattr(self.model, "_features", None) or FEATURES)
        if len(self.features) != int(self.model.cfg["n_feat"]):
            raise ValueError(f"特征名数 {len(self.features)} 与模型 n_feat "
                             f"{self.model.cfg['n_feat']} 不一致")
        self.seq_len = int(self.model.cfg.get("seq_len") or seq_len)
        self.buf = SeqFeatureBuffer(self.seq_len, vel_w, dt)

    def push(self, *a, **kw) -> None:
        self.buf.push(*a, **kw)

    def ready(self) -> bool:
        return self.buf.ready()

    def reset(self) -> None:
        self.buf.reset()

    def predict(self):
        """→ (du_fb, du_lr)；缓冲未满/推理异常返回 None（调用方退回静态基座）。"""
        win = self.buf.window()
        if win is None:
            return None
        try:
            import torch
            dev = next(self.model.parameters()).device
            with torch.no_grad():
                y = self.model(torch.as_tensor(win[None, ...], device=dev))[0].cpu().numpy()
        except Exception:                      # 推理异常绝不能影响控制回路
            return None
        return float(y[0]), float(y[1])


def load_ref(path):
    """读配置 JSON 的 dynamic_seq 段 → (模型绝对路径, 段 dict)。"""
    ref = json.loads(Path(path).read_text(encoding="utf-8-sig"))["dynamic_seq"]
    p = Path(ref["model"])
    if not p.is_absolute():
        p = Path(path).parent / p
    return p, ref
