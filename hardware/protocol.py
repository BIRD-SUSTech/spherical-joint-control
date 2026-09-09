"""舵机串口协议 v2：单帧多通道（原子发布）。

设计目标（设计文档 §8.3.2 + 帧间同步问题）：
    旧协议每拍发 2~4 个 5 字节帧（id=1/2/3/4），帧间 ~434µs 串行到达，固件逐帧立即
    生效 → 出现"半更新命令"（差分已是新值、共模还是旧值）。v2 改为【单帧 4 通道】，
    固件一次收齐、原子写入 4 个舵机，彻底消除帧间不同步。

帧布局（固定 11 字节，固件固定读 11 字节）：
    byte0       HEAD        0xAA
    byte1       CMD         0x00=RELAX / 0x01=WRITE
    byte2-3     CH1         int16 big-endian   # 前后对"正侧"
    byte4-5     CH2         int16 big-endian   # 左右对"正侧"
    byte6-7     CH3         int16 big-endian   # 前后对"负侧"
    byte8-9     CH4         int16 big-endian   # 左右对"负侧"
    byte10      TAIL        0x55

    RELAX 帧：CMD=0x00，8 字节载荷建议发 0（固件忽略载荷，只执行放松/急停）。
    WRITE 帧：CMD=0x01，固件把 CH1..CH4 一次性写入 4 个舵机（原子生效）。

通道语义（host 侧唯一真值源；固件【无需理解】语义，只按顺序写 CH1..CH4）：
    CH1 = +u_fb + c_fb       前后对正侧
    CH3 = -u_fb + c_fb       前后对负侧
    CH2 = +u_lr + c_lr       左右对正侧
    CH4 = -u_lr + c_lr       左右对负侧
    u_fb / u_lr = 差分 offset（产扭矩，闭环跟踪量）
    c_fb / c_lr = 共模预紧 offset（只增刚度、不产净扭矩，§8.3.2）
    由 host 在 mix_channels() 里合成并逐缆夹紧到 int16。

固件实现要点（实机 agent 照此改）：
    1. 固定读 11 字节；校验 HEAD/TAIL，按 CMD 分发；未知 CMD 忽略整帧。
    2. CMD=0x01：把 4 个 int16 分别写入 CH1..CH4 四个舵机，【一次性生效】——
       不要在循环里逐个写（否则又退化成帧内不同步）。
    3. 每通道在【最终值】上独立夹紧到物理限位；不要在中间量（差分/共模）上夹。
    4. CMD=0x00：执行放松（原 id=0 语义），忽略 8 字节载荷。

offset 不做固有限幅（设计文档 T1：安全界在关节角空间 65°，见 excite.guardian）。
host 仅夹紧到 int16 范围，防止 struct.pack 溢出崩溃。
"""

from __future__ import annotations

import struct

HEAD = 0xAA
TAIL = 0x55
CMD_RELAX = 0x00
CMD_WRITE = 0x01

INT16_MIN = -(2**15)
INT16_MAX = 2**15 - 1

# HEAD(1B) CMD(1B) CH1..CH4(4×2B) TAIL(1B) = 11 字节
_FRAME = struct.Struct(">BBhhhhB")


def _clamp(v) -> int:
    v = int(v)
    return max(INT16_MIN, min(INT16_MAX, v))


def mix_channels(u_fb, u_lr, c_fb=0, c_lr=0) -> tuple[int, int, int, int]:
    """差分 + 共模 → 4 个每缆 offset（host 语义，含逐缆夹紧）。

    CH1 = +u_fb + c_fb    CH3 = -u_fb + c_fb
    CH2 = +u_lr + c_lr    CH4 = -u_lr + c_lr
    """
    return (
        _clamp(u_fb + c_fb),
        _clamp(u_lr + c_lr),
        _clamp(-u_fb + c_fb),
        _clamp(-u_lr + c_lr),
    )


def encode_write(ch1, ch2, ch3, ch4) -> bytes:
    """编码 WRITE 帧（4 通道原子写）。"""
    return _FRAME.pack(HEAD, CMD_WRITE, _clamp(ch1), _clamp(ch2), _clamp(ch3), _clamp(ch4), TAIL)


def encode_relax() -> bytes:
    """编码 RELAX 帧（放松/急停，载荷置 0）。"""
    return _FRAME.pack(HEAD, CMD_RELAX, 0, 0, 0, 0, TAIL)
