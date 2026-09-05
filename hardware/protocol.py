"""舵机串口协议：唯一帧格式 ``0xAA | id(1B) | int16_be(2B) | 0x55``。

物理命名约定（设计文档 §6，唯一真值源）：
    FRONT_BACK_ID = 1   前后对（固件 Coupled_Servo(CH3, CH1)）
    LEFT_RIGHT_ID = 2   左右对（固件 Coupled_Servo(CH4, CH2)）
    RELAX_ID      = 0   放松（四路松缆，安全急停）

主机只发 2 路差分 offset；对侧耦合在固件内完成（一端 +offset、另一端 -offset）。
offset 不做固有限幅（设计文档 T1：安全界在关节角空间 65°，见 excite.guardian）。
此处仅夹紧到 int16 范围，防止 struct.pack 溢出崩溃。
"""

from __future__ import annotations

import struct

FRONT_BACK_ID = 1
LEFT_RIGHT_ID = 2
RELAX_ID = 0

HEAD = 0xAA
TAIL = 0x55
INT16_MIN = -(2**15)
INT16_MAX = 2**15 - 1

_FRAME = struct.Struct(">BBhB")


def encode_frame(servo_id: int, offset: int) -> bytes:
    """把 (id, offset) 编码为 5 字节帧。offset 先夹紧到 int16。"""
    offset = int(offset)
    offset = max(INT16_MIN, min(INT16_MAX, offset))
    return _FRAME.pack(HEAD, servo_id, offset, TAIL)


def relax_frame() -> bytes:
    """放松帧（id=0），触发固件松缆。"""
    return encode_frame(RELAX_ID, 0)
