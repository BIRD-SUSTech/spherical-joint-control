"""时序动态前馈模型（torch 单一实现）。

**为什么必须用 torch 而不是手写 numpy 前向**：训练 / 导出 / 运行时必须是**同一份代码**。
手写 numpy 前向与训练实现极易漂移（GRU 门顺序、偏置布局 `[r|z|n]`、归一化位置），
且完全用不上 CUDA。这里把归一化折成 `register_buffer`，模型直接吃**原始单位**特征，
运行时一行 `model(x)`。

    输入  x : (B, L, n_feat) 原始单位        输出  Δu : (B, n_out)  offset

    fit()       训练（device="auto" → cuda 可用即用）
    save_model() / load_model()   单文件存取（cfg + state_dict + 特征名 + meta）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def pick_device(pref: str = "auto") -> torch.device:
    """auto → cuda（可用时）否则 cpu；也可显式 'cuda' / 'cuda:1' / 'cpu'。"""
    if pref and pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqFeedforward(nn.Module):
    """GRU 时序编码器 + 线性头；归一化折进 buffer（运行时喂原始单位）。"""

    def __init__(self, n_feat: int, hidden: int = 32, n_out: int = 2, seq_len: int = 10):
        super().__init__()
        self.cfg = {"n_feat": int(n_feat), "hidden": int(hidden),
                    "n_out": int(n_out), "seq_len": int(seq_len)}
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)
        self.register_buffer("in_mean", torch.zeros(n_feat))
        self.register_buffer("in_std", torch.ones(n_feat))
        self.register_buffer("out_mean", torch.zeros(n_out))
        self.register_buffer("out_std", torch.ones(n_out))

    def set_norm(self, in_mean, in_std, out_mean, out_std) -> None:
        with torch.no_grad():
            self.in_mean.copy_(torch.as_tensor(in_mean, dtype=torch.float32))
            self.in_std.copy_(torch.as_tensor(in_std, dtype=torch.float32))
            self.out_mean.copy_(torch.as_tensor(out_mean, dtype=torch.float32))
            self.out_std.copy_(torch.as_tensor(out_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,L,n_feat) 原始单位 → Δu (B,n_out)；也接受 (L,n_feat)（自动加 batch 维）。"""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        h, _ = self.gru((x - self.in_mean) / self.in_std)
        y = self.head(h[:, -1]) * self.out_std + self.out_mean
        return y.squeeze(0) if squeeze else y

    @torch.no_grad()
    def predict_np(self, x_np) -> np.ndarray:
        """numpy 进 / numpy 出（评估与运行时用）；内部仍是 torch 前向。"""
        dev = next(self.parameters()).device
        x = torch.as_tensor(np.asarray(x_np, dtype=np.float32), device=dev)
        return self(x).cpu().numpy()


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def fit(W, Y, hidden=32, epochs=60, lr=2e-3, batch=256, seed=0, val_frac=0.15,
        device="auto", val=None, verbose=True):
    """训练 SeqFeedforward；W (m,L,F) 原始单位，Y (m,2)。

    val=(W_va, Y_va) 时用外部留出做早停/选优（LOSO 协议）；否则从训练集随机切 val_frac。
    返回 (model, history)；model 已载入最优权重并置于 **cpu**（便于导出与逐位复现）。
    """
    dev = pick_device(device)
    torch.manual_seed(seed)
    m, L, F = W.shape
    if val is None:
        nv = max(256, int(val_frac * m))
        perm = np.random.default_rng(seed).permutation(m)
        va, tr = perm[:nv], perm[nv:]
        Wva, Yva = W[va], Y[va]
    else:
        tr = np.arange(m)
        Wva, Yva = val
    Wtr, Ytr = W[tr], Y[tr]

    model = SeqFeedforward(F, hidden, Y.shape[1], L)
    model.set_norm(Wtr.reshape(-1, F).mean(0), Wtr.reshape(-1, F).std(0) + 1e-8,
                   Ytr.mean(0), Ytr.std(0) + 1e-8)
    model.to(dev)

    xt = torch.as_tensor(Wtr, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(np.asarray(Ytr), dtype=torch.float32, device=dev)
    xv = torch.as_tensor(np.asarray(Wva), dtype=torch.float32, device=dev)
    yv = torch.as_tensor(np.asarray(Yva), dtype=torch.float32, device=dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best, best_state, hist = float("inf"), None, []
    for ep in range(epochs):
        model.train()
        idx = np.random.default_rng(seed + ep).permutation(len(tr))
        for b in range(0, len(idx), batch):
            sel = torch.as_tensor(idx[b:b + batch], dtype=torch.long, device=dev)
            if len(sel) < 8:
                continue
            opt.zero_grad()
            lossf(model(xt[sel]), yt[sel]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(xv), yv).item())
        hist.append(vl)
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"    epoch {ep+1:3d}  val MSE(标准) {vl:.5f}")
    if best_state:
        model.load_state_dict(best_state)
    return model.cpu().eval(), hist


# ---------------------------------------------------------------------------
# 存取：torch.save 单文件（cfg + state_dict + 特征名 + meta）
# ---------------------------------------------------------------------------

def save_model(model: SeqFeedforward, path, features=None, meta=None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": model.cfg, "state_dict": model.state_dict(),
                "features": [str(f) for f in (features or [])], "meta": dict(meta or {})}, p)
    return p


def load_model(path, device="cpu") -> SeqFeedforward:
    """载入 → SeqFeedforward（eval 模式）。运行时缺省 cpu（控制回路不需 GPU）。"""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    model = SeqFeedforward(cfg["n_feat"], cfg["hidden"], cfg["n_out"], cfg["seq_len"])
    model.load_state_dict(blob["state_dict"])
    model.to(pick_device(device)).eval()
    model._features = list(blob.get("features") or [])
    model._meta = dict(blob.get("meta") or {})
    return model


def model_info(path) -> dict:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {"cfg": blob["cfg"], "features": list(blob.get("features") or []),
            "meta": dict(blob.get("meta") or {})}
