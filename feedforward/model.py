"""前向模型：MLP 学 (q 历史, u 历史) -> Δq。"""

from __future__ import annotations

from torch import nn


class ForwardMLP(nn.Module):
    def __init__(self, in_dim=12, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
