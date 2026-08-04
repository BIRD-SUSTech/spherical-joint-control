"""IM948 BLE IMU 采集器。

使用 asyncio-in-thread 模式：bleak 需要在 asyncio event loop 中运行，
因此在一个独立 daemon 线程中执行 asyncio.run()。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time

from bleak import BleakClient, BleakScanner

from ..config import ImuConfig
from ..utils.data_types import ImuPacket

logger = logging.getLogger(__name__)


class ImuCollector:
    """IM948 BLE 采集器。内部运行 asyncio event loop。"""

    def __init__(
        self,
        config: ImuConfig,
        output_queue: queue.Queue,
        start_event: threading.Event,
        stop_event: threading.Event,
    ):
        self._config = config
        self._output_queue = output_queue
        self._start_event = start_event
        self._stop_event = stop_event
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_stop: asyncio.Event | None = None
        self._dropped_count = 0
        self._row_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_asyncio_loop, name="ImuCollector", daemon=True
        )
        self._thread.start()
        logger.info("IMU collector thread started")

    def stop(self) -> None:
        if self._loop and self._async_stop:
            self._loop.call_soon_threadsafe(self._async_stop.set)
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def row_count(self) -> int:
        return self._row_count

    # ------------------------------------------------------------------
    def _run_asyncio_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception:
            logger.exception("IMU asyncio loop crashed")
        finally:
            self._loop.close()

    async def _async_main(self) -> None:
        self._async_stop = asyncio.Event()
        config = self._config

        device = await self._find_device()
        if device is None:
            logger.error("IM948 not found")
            return

        logger.info("Connecting to IM948 [%s]...", device.address)

        def on_disconnect(_client):
            logger.warning("IM948 disconnected")
            if self._async_stop:
                self._loop.call_soon_threadsafe(self._async_stop.set)

        async with BleakClient(device, disconnected_callback=on_disconnect) as client:
            logger.info("IM948 connected")

            await client.start_notify(
                config.notify_characteristic,
                lambda _char, data: self._on_notification(data),
            )

            # 保持连接
            await client.write_gatt_char(config.write_characteristic, bytes([0x29]))
            await asyncio.sleep(0.2)
            # 高速通信
            await client.write_gatt_char(config.write_characteristic, bytes([0x46]))
            await asyncio.sleep(0.2)

            # 设置参数
            params = bytearray(11)
            params[0] = 0x12
            params[1] = 5
            params[2] = 255
            params[3] = 0
            params[4] = ((config.barometer_filter & 3) << 1) | (config.is_compass_on & 1)
            params[5] = config.report_hz
            params[6] = 1
            params[7] = 3
            params[8] = 5
            params[9] = config.report_tag & 0xFF
            params[10] = (config.report_tag >> 8) & 0xFF
            await client.write_gatt_char(config.write_characteristic, params)
            await asyncio.sleep(0.2)

            # 开启主动上报
            await client.write_gatt_char(config.write_characteristic, bytes([0x19]))

            # 等待姿态稳定后坐标系清零
            await asyncio.sleep(2.0)
            await client.write_gatt_char(config.write_characteristic, bytes([0x06]))
            await asyncio.sleep(0.3)

            logger.info("IM948 logging started at %d Hz", config.report_hz)

            # 等待停止信号
            await self._async_stop.wait()

            # 关闭主动上报
            await client.write_gatt_char(config.write_characteristic, bytes([0x18]))
            await client.stop_notify(config.notify_characteristic)

        logger.info("IM948 stopped, total rows: %d", self._row_count)

    async def _find_device(self):
        config = self._config
        logger.info("Scanning BLE for IM948...")

        device = await BleakScanner.find_device_by_address(
            config.device_address, cb=dict(use_bdaddr=False), timeout=8.0
        )
        if device:
            return device

        return await BleakScanner.find_device_by_filter(
            lambda d, ad: config.device_name_keyword
            in ((d.name or ad.local_name or "").lower()),
            timeout=config.scan_timeout_s,
        )

    def _on_notification(self, data: bytes) -> None:
        packet = ImuPacket.from_raw_packet(data)
        if packet is None:
            return
        self._row_count += 1
        try:
            self._output_queue.put_nowait(packet)
        except queue.Full:
            self._dropped_count += 1
