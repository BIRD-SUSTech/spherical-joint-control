"""力传感器 MODBUS-RTU 采集线程（提炼自 archive/data_collection/sensor_collectors/force_collector.py）。

通道映射（新硬件 rig2 确认）：ch1–ch4 = 4 根缆的张力（共模预紧标定 §8.3.2 只用这 4 个），
ch5/ch6 不用。ch1..ch4 与协议 CH1..CH4（4 个舵机）的【物理缆对应关系】在 M3 力标定时
用单缆阶跃实测确认（不猜测）。

速率（提速后默认）：baudrate=**19200（不改）**、channel_count=4、**sample_interval_ms=0**
    → 周期 = max(读取耗时, 间隔)，间隔 0 = 读多快就多快（受传感器响应封顶）。
    读取耗时估算：4 通道 8 寄存器，请求 8B + 响应 21B = 29B @19200 ≈ **15ms** 线时间
    + 传感器响应延迟 → 预估 **~30-60 Hz**（原实现 6ch/100ms 附加 sleep：实测 8.25 Hz、间隔 121ms）。
    采集后请核对 `session_metadata.json` 的 `force.rate_hz` 与 `force.error_count`
    判断是否真的提上去了 / 是否有通信错误。
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
        channel_count: int = 4,
        scale_factor: float = 1.0,
        sample_interval_ms: int = 0,
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
        self._sample_count = 0
        self._t_first = None
        self._t_last = None

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

    @property
    def rate_hz(self) -> float:
        """实测平均采样率（按首末样本的 wall-clock 间隔算）。"""
        if self._t_first is None or self._t_last is None or self._t_last <= self._t_first:
            return 0.0
        return (self._sample_count - 1) / (self._t_last - self._t_first)

    def _loop(self) -> None:
        from .modbus import ModbusException

        # 定周期调度：周期 = max(读取耗时, interval_s)；interval=0 → 读多快就多快（传感器极限）。
        # （旧实现是"读耗时 + sleep(interval)"，固定 50ms 下限会白白吃掉一半带宽。）
        interval_s = max(0.0, self._sample_interval_ms / 1000.0)
        next_t = time.perf_counter()
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
                now = time.perf_counter()
                if self._t_first is None:
                    self._t_first = now
                self._t_last = now
                self._sample_count += 1
                try:
                    self._output_queue.put_nowait(data)
                except queue.Full:
                    self._dropped_count += 1
            except ModbusException:
                self._error_count += 1
                time.sleep(0.05)
                next_t = time.perf_counter()  # 出错后重新对齐周期
                continue
            next_t += interval_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()  # 读取本身已超周期，重新对齐（不累积相位）
