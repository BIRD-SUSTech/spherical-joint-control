"""力传感器 MODBUS-RTU 采集线程（提炼自 archive/data_collection/sensor_collectors/force_collector.py）。

通道映射（新硬件 rig2 确认）：ch1–ch4 = 4 根缆的张力（共模预紧标定 §8.3.2 只用这 4 个），
ch5/ch6 不用。ch1..ch4 与协议 CH1..CH4（4 个舵机）的【物理缆对应关系】在 M3 力标定时
用单缆阶跃实测确认（不猜测）。
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from ..schema import ForceData

logger = logging.getLogger(__name__)


class ForceCollector:
    """独立 daemon 线程周期性读取六维力传感器。"""

    def __init__(
        self,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        serial_port: str,
        baudrate: int = 19200,
        data_bits: int = 8,
        stop_bits: int = 1,
        parity: str = "N",
        timeout: float = 0.5,
        rs485_mode: bool = False,
        rs485_tx_pin: str = "RTS",
        rs485_tx_level: bool = True,
        slave_address: int = 0x01,
        reg_start_address: int = 0x000B,
        channel_count: int = 6,
        scale_factor: float = 1.0,
        sample_interval_ms: int = 100,
    ):
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._serial_port = serial_port
        self._baudrate = baudrate
        self._data_bits = data_bits
        self._stop_bits = stop_bits
        self._parity = parity
        self._timeout = timeout
        self._rs485_mode = rs485_mode
        self._rs485_tx_pin = rs485_tx_pin
        self._rs485_tx_level = rs485_tx_level
        self._slave_address = slave_address
        self._reg_start_address = reg_start_address
        self._channel_count = channel_count
        self._scale_factor = scale_factor
        self._sample_interval_ms = sample_interval_ms
        self._thread = None
        self._client = None
        self._dropped_count = 0
        self._error_count = 0

    def start(self) -> None:
        from .modbus import ModbusClient  # 惰性，mock 不依赖 pyserial

        self._client = ModbusClient(
            port=self._serial_port,
            baudrate=self._baudrate,
            data_bits=self._data_bits,
            stop_bits=self._stop_bits,
            parity=self._parity,
            timeout=self._timeout,
            rs485_mode=self._rs485_mode,
            rs485_tx_pin=self._rs485_tx_pin,
            rs485_tx_level=self._rs485_tx_level,
        )
        self._thread = threading.Thread(target=self._loop, name="ForceCollector", daemon=True)
        self._thread.start()
        logger.info("Force collector started on %s (slave=0x%02X)",
                    self._serial_port, self._slave_address)

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._close()

    def _close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def _loop(self) -> None:
        from .modbus import ModbusException

        interval_s = max(0.05, self._sample_interval_ms / 1000.0)
        while not self._stop_event.is_set():
            try:
                raw = self._client.read_32bit_values(
                    self._slave_address, self._reg_start_address,
                    self._channel_count, signed=True,
                )
                values = [v * self._scale_factor for v in raw]
                padded = values + [0.0] * (6 - len(values))
                data = ForceData(
                    pc_timestamp_ns=time.perf_counter_ns(),
                    pc_receive_unix_time_ms=int(time.time() * 1000),
                    ch1=padded[0], ch2=padded[1], ch3=padded[2],
                    ch4=padded[3], ch5=padded[4], ch6=padded[5],
                )
                try:
                    self._output_queue.put_nowait(data)
                except queue.Full:
                    self._dropped_count += 1
            except ModbusException:
                self._error_count += 1
                time.sleep(0.05)
                continue
            time.sleep(interval_s)
