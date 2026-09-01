"""开环激励（已迁移到 control_model.excitation，单一数据源）。

data_collection 与 feedforward 共用同一份实现；此处保留重导出以兼容旧引用。
"""

from control_model.excitation import (
    circle,
    d_from_joint_deg,
    decode,
    default_segments,
    lissajous,
    sample_segment,
)

__all__ = [
    "decode", "d_from_joint_deg", "lissajous", "circle",
    "default_segments", "sample_segment",
]
