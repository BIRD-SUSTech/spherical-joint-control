"""前馈补偿模型 g：MLP 学 (参考目标, 实测姿态, 传感器) -> 差分指令。

输出为差分 d=(d1,d2)（2 维），解码时叠加固定共模预紧 p 得到 4 路舵机角，
规避 4->2 冗余病态。
"""

from __future__ import annotations

from torch import nn


class FeedforwardMLP(nn.Module):
    def __init__(self, in_dim=13, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
