"""舵机总线：串口驱动 + 单帧多通道发布（差分 + 共模预紧，原子生效）。

协议 v2（见 hardware/protocol.py）：单帧 4 通道，固件一次收齐、原子写入 4 个舵机。
从 archive/example_code/steering_motor 的协议升级而来。真实总线走串口（115200），
Mock 总线用于 --mock/--dry-run 离线自检。
"""

from __future__ import annotations

import logging

from . import protocol

logger = logging.getLogger(__name__)


class ServoBus:
    """真实串口舵机总线。"""

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._ser = None

    def connect(self) -> bool:
        import serial

        try:
            self._ser = serial.Serial(self.port, self.baudrate,
                                      timeout=0.1, write_timeout=0.1)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("串口 %s 打开失败: %s", self.port, e)
            return False

    def _write(self, frame: bytes) -> None:
        if self._ser and self._ser.is_open:
            self._ser.write(frame)

    def send_channels(self, ch1: int, ch2: int, ch3: int, ch4: int) -> None:
        """原子写 4 个每缆 offset（单帧 11 字节）。"""
        self._write(protocol.encode_write(ch1, ch2, ch3, ch4))

    def send_pair(self, front_back: int, left_right: int) -> None:
        """发 2 路差分 offset（共模 = 0，向后兼容）。"""
        self.send_channels(*protocol.mix_channels(front_back, left_right))

    def send_pair_tension(self, front_back: int, left_right: int,
                          c_front_back: int, c_left_right: int) -> None:
        """原子发差分 + 共模预紧（设计文档 §8.3.2）。

        差分 u=(front_back, left_right) 产扭矩；共模 c=(c_front_back, c_left_right)
        只增刚度。两者由 mix_channels 合成 4 个每缆值，单帧一次性发布，无帧间不同步。
        """
        self.send_channels(*protocol.mix_channels(front_back, left_right,
                                                  c_front_back, c_left_right))

    def relax(self) -> None:
        """放松（CMD_RELAX），安全急停。"""
        self._write(protocol.encode_relax())

    def close(self) -> None:
        if self._ser:
            self._ser.close()
            self._ser = None


class MockServoBus:
    """不接串口时的总线（--mock/--dry-run）：只记录指令。"""

    def connect(self) -> bool:
        return True

    def send_channels(self, ch1: int, ch2: int, ch3: int, ch4: int) -> None:
        logger.info("cmd CH1=%d CH2=%d CH3=%d CH4=%d", ch1, ch2, ch3, ch4)

    def send_pair(self, front_back: int, left_right: int) -> None:
        logger.info("cmd fb=%d lr=%d", front_back, left_right)

    def send_pair_tension(self, front_back: int, left_right: int,
                          c_front_back: int, c_left_right: int) -> None:
        logger.info("cmd fb=%d lr=%d tension_fb=%d tension_lr=%d",
                    front_back, left_right, c_front_back, c_left_right)

    def relax(self) -> None:
        logger.info("cmd relax")

    def close(self) -> None:
        pass
