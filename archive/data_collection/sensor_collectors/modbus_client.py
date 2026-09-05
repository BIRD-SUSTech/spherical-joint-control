"""
MODBUS-RTU 协议客户端 (RS232/RS485)。
支持功能码 0x03 (读保持寄存器) 和 0x10 (写多个寄存器)。
"""

import struct
import time
import serial


# --- CRC-16 ---

def _build_crc_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
        table.append(crc)
    return tuple(table)


_MODBUS_CRC_TABLE = _build_crc_table()


def calc_crc(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        idx = (crc ^ byte) & 0xFF
        crc = ((crc >> 8) ^ _MODBUS_CRC_TABLE[idx]) & 0xFFFF
    return crc


def _wrap_crc(data: bytes) -> bytes:
    crc = calc_crc(data)
    return data + struct.pack("<H", crc)


# --- 异常码 ---
_EXCEPTION_CODES = {
    0x01: "非法功能码",
    0x02: "非法数据地址",
    0x03: "非法数据值",
    0x04: "从机设备故障",
}


class ModbusException(Exception):
    def __init__(self, code: int, message: str = ""):
        self.code = code
        desc = _EXCEPTION_CODES.get(code, "未知异常")
        super().__init__(f"MODBUS 异常 0x{code:02X}: {desc}" + (f" — {message}" if message else ""))


class ModbusClient:
    """MODBUS-RTU 串行通讯客户端。"""

    def __init__(
        self,
        port: str,
        baudrate: int = 19200,
        data_bits: int = 8,
        stop_bits: int = 1,
        parity: str = "N",
        timeout: float = 0.5,
        rs485_mode: bool = False,
        rs485_tx_pin: str = "RTS",
        rs485_tx_level: bool = True,
        debug: bool = False,
    ):
        self._debug = debug
        self._rs485 = rs485_mode
        self._tx_pin = rs485_tx_pin.upper()
        self._tx_level = rs485_tx_level
        parity_map = {"N": serial.PARITY_NONE, "E": serial.PARITY_EVEN, "O": serial.PARITY_ODD}
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=data_bits,
            stopbits=serial.STOPBITS_ONE if stop_bits == 1 else serial.STOPBITS_TWO,
            parity=parity_map.get(parity.upper(), serial.PARITY_NONE),
            timeout=timeout,
        )
        self._timeout = timeout
        if self._rs485:
            self._set_tx(False)

    def close(self):
        self._ser.close()

    @property
    def is_open(self) -> bool:
        return self._ser.is_open

    def _set_tx(self, tx: bool):
        """设置 RS485 收发模式。"""
        level = self._tx_level if tx else not self._tx_level
        if self._tx_pin == "DTR":
            self._ser.dtr = level
        else:
            self._ser.rts = level

    def _send(self, slave: int, func: int, data: bytes) -> bytes:
        frame = _wrap_crc(struct.pack("BB", slave, func) + data)
        if self._debug:
            print(f"[DEBUG] TX ({len(frame)}B): {frame.hex(' ')}")

        self._ser.reset_input_buffer()

        if self._rs485:
            self._set_tx(True)       # 切换为发送模式
            self._ser.write(frame)
            self._ser.flush()        # 确保数据完全发出
            self._set_tx(False)      # 切回接收模式
        else:
            self._ser.write(frame)
            time.sleep(0.02)

        # 读取从机地址 + 功能码
        header = self._ser.read(2)
        if self._debug:
            print(f"[DEBUG] RX header: {header.hex(' ') if header else '(空)'}")
        if len(header) < 2:
            raise ModbusException(0, "无响应 (超时)")

        rsp_slave, rsp_func = struct.unpack("BB", header)

        if rsp_func & 0x80:
            exc_code = self._ser.read(1)
            if len(exc_code) < 1:
                raise ModbusException(0, "异常响应不完整")
            raise ModbusException(exc_code[0])

        if rsp_func == 0x03:
            byte_count = self._ser.read(1)
            if len(byte_count) < 1:
                raise ModbusException(0, "响应不完整")
            payload = bytes(byte_count) + self._ser.read(byte_count[0])
        elif rsp_func == 0x10:
            payload = self._ser.read(4)
        else:
            payload = self._ser.read(256)

        crc_bytes = self._ser.read(2)
        if len(crc_bytes) < 2:
            raise ModbusException(0, "CRC 不完整")

        full_rsp = bytes(header) + payload + crc_bytes
        if calc_crc(full_rsp) != 0:
            raise ModbusException(0, "CRC 校验失败")

        return payload

    def read_holding_registers(self, slave: int, start_addr: int, count: int) -> list[int]:
        """功能码 0x03 — 读保持寄存器。"""
        if not (1 <= count <= 125):
            raise ValueError("寄存器数量须在 1-125 之间")
        data = struct.pack(">HH", start_addr, count)
        payload = self._send(slave, 0x03, data)
        byte_count = payload[0]
        reg_data = payload[1:]
        fmt = f">{count}H"
        return list(struct.unpack(fmt, reg_data[: byte_count]))

    def write_multiple_registers(self, slave: int, start_addr: int, values: list[int]):
        """功能码 0x10 — 写多个寄存器。"""
        count = len(values)
        if not (1 <= count <= 123):
            raise ValueError("寄存器数量须在 1-123 之间")
        byte_count = count * 2
        data = struct.pack(f">HHB{count}H", start_addr, count, byte_count, *values)
        self._send(slave, 0x10, data)

    def read_32bit_values(self, slave: int, start_addr: int, count: int, signed: bool = True) -> list[int]:
        """读取 32 位值 (每值占 2 个保持寄存器)。"""
        regs = self.read_holding_registers(slave, start_addr, count * 2)
        fmt_char = "i" if signed else "I"
        values = []
        for i in range(0, len(regs), 2):
            raw = (regs[i] << 16) | regs[i + 1]
            values.append(struct.unpack(f">{fmt_char}", struct.pack(">I", raw))[0])
        return values

    def write_32bit_values(self, slave: int, start_addr: int, values: list[int]):
        """写入 32 位值 (每值占 2 个保持寄存器)。"""
        regs = []
        for v in values:
            raw = struct.unpack(">I", struct.pack(">i", v))[0]
            regs.extend([(raw >> 16) & 0xFFFF, raw & 0xFFFF])
        self.write_multiple_registers(slave, start_addr, regs)
