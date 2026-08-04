"""力传感器 MODBUS-RTU 采集线程。"""

from __future__ import annotations

import logging
import queue
import threading
import time

from ..config import ForceConfig
from ..utils.data_types import ForceData
from .modbus_client import ModbusClient, ModbusException

logger = logging.getLogger(__name__)


class ForceCollector:
    """在独立 daemon 线程中周期性读取六通道力传感器。"""

    def __init__(
        self,
        config: ForceConfig,
        output_queue: queue.Queue,
        start_event: threading.Event,
        stop_event: threading.Event,
    ):
        self._config = config
        self._output_queue = output_queue
        self._start_event = start_event
        self._stop_event = stop_event
        self._thread: threading.Thread | None = None
        self._client: ModbusClient | None = None
        self._dropped_count = 0
        self._error_count = 0

    def start(self) -> None:
        self._client = ModbusClient(
            port=self._config.serial_port,
            baudrate=self._config.baudrate,
            data_bits=self._config.data_bits,
            stop_bits=self._config.stop_bits,
            parity=self._config.parity,
            timeout=self._config.timeout,
            rs485_mode=self._config.rs485_mode,
            rs485_tx_pin=self._config.rs485_tx_pin,
            rs485_tx_level=self._config.rs485_tx_level,
        )
        self._thread = threading.Thread(
            target=self._loop, name="ForceCollector", daemon=True
        )
        self._thread.start()
        logger.info(
            "Force collector started on %s (slave=0x%02X)",
            self._config.serial_port,
            self._config.slave_address,
        )

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

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        interval_s = max(0.05, self._config.sample_interval_ms / 1000.0)
        slave = self._config.slave_address
        start_addr = self._config.reg_start_address
        channel_count = self._config.channel_count
        scale = self._config.scale_factor

        while not self._stop_event.is_set():
            try:
                raw = self._client.read_32bit_values(slave, start_addr, channel_count, signed=True)
                values = [v * scale for v in raw]
                data = ForceData(
                    pc_timestamp_ns=time.perf_counter_ns(),
                    pc_receive_unix_time_ms=int(time.time() * 1000),
                    ch1=values[0], ch2=values[1], ch3=values[2],
                    ch4=values[3], ch5=values[4], ch6=values[5],
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
