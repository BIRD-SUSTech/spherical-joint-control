"""舵机总线：串口驱动 + 发送 2 路差分 offset + 放松/急停。

从 archive/example_code/steering_motor 的协议提炼。真实总线走串口（COM5, 115200），
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

    def send(self, servo_id: int, offset: int) -> None:
        if self._ser and self._ser.is_open:
            self._ser.write(protocol.encode_frame(servo_id, offset))

    def send_pair(self, front_back: int, left_right: int) -> None:
        """发送 2 路差分 offset（前后、左右）。"""
        self.send(protocol.FRONT_BACK_ID, front_back)
        self.send(protocol.LEFT_RIGHT_ID, left_right)

    def relax(self) -> None:
        """松缆（id=0），安全急停。"""
        self.send(protocol.RELAX_ID, 0)

    def close(self) -> None:
        if self._ser:
            self._ser.close()
            self._ser = None


class MockServoBus:
    """不接串口时的总线（--mock/--dry-run）：只记录指令。"""

    def connect(self) -> bool:
        return True

    def send(self, servo_id: int, offset: int) -> None:
        logger.info("cmd id=%d offset=%d", servo_id, offset)

    def send_pair(self, front_back: int, left_right: int) -> None:
        logger.info("cmd fb=%d lr=%d", front_back, left_right)

    def relax(self) -> None:
        logger.info("cmd relax")

    def close(self) -> None:
        pass
