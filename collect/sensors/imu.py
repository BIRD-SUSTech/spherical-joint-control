"""IM948 BLE IMU 采集线程（提炼自 archive/data_collection/sensor_collectors/imu_collector.py）。

asyncio-in-thread 模式：bleak 需要 asyncio event loop，在独立 daemon 线程中 run_until_complete。
bleak 惰性导入，离线 mock 冒烟不依赖。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

from ..schema import ImuPacket

logger = logging.getLogger(__name__)


class ImuCollector:
    def __init__(
        self,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        device_address: str = "A5:B2:90:FF:4A:12",
        device_name_keyword: str = "im948",
        report_hz: int = 100,
        report_tag: int = 0x0025,
        notify_characteristic: int = 0x0007,
        write_characteristic: int = 0x0005,
        scan_timeout_s: float = 10.0,
        is_compass_on: int = 0,
        barometer_filter: int = 2,
    ):
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._device_address = device_address
        self._device_name_keyword = device_name_keyword
        self._report_hz = report_hz
        self._report_tag = report_tag
        self._notify_char = notify_characteristic
        self._write_char = write_characteristic
        self._scan_timeout_s = scan_timeout_s
        self._is_compass_on = is_compass_on
        self._barometer_filter = barometer_filter

        self._thread = None
        self._loop = None
        self._async_stop = None
        self._dropped_count = 0
        self._row_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_asyncio_loop,
                                        name="ImuCollector", daemon=True)
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
        from bleak import BleakClient, BleakScanner  # 惰性

        self._async_stop = asyncio.Event()
        device = await self._find_device(BleakScanner)
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
                self._notify_char,
                lambda _char, data: self._on_notification(data),
            )
            await client.write_gatt_char(self._write_char, bytes([0x29]))
            await asyncio.sleep(0.2)
            await client.write_gatt_char(self._write_char, bytes([0x46]))
            await asyncio.sleep(0.2)

            params = bytearray(11)
            params[0] = 0x12
            params[1] = 5
            params[2] = 255
            params[3] = 0
            params[4] = ((self._barometer_filter & 3) << 1) | (self._is_compass_on & 1)
            params[5] = self._report_hz
            params[6] = 1
            params[7] = 3
            params[8] = 5
            params[9] = self._report_tag & 0xFF
            params[10] = (self._report_tag >> 8) & 0xFF
            await client.write_gatt_char(self._write_char, params)
            await asyncio.sleep(0.2)
            await client.write_gatt_char(self._write_char, bytes([0x19]))

            await asyncio.sleep(2.0)
            await client.write_gatt_char(self._write_char, bytes([0x06]))
            await asyncio.sleep(0.3)

            logger.info("IM948 logging started at %d Hz", self._report_hz)
            await self._async_stop.wait()

            await client.write_gatt_char(self._write_char, bytes([0x18]))
            await client.stop_notify(self._notify_char)

        logger.info("IM948 stopped, total rows: %d", self._row_count)

    async def _find_device(self, BleakScanner):
        logger.info("Scanning BLE for IM948...")
        device = await BleakScanner.find_device_by_address(
            self._device_address, cb=dict(use_bdaddr=False), timeout=8.0
        )
        if device:
            return device
        return await BleakScanner.find_device_by_filter(
            lambda d, ad: self._device_name_keyword in ((d.name or ad.local_name or "").lower()),
            timeout=self._scan_timeout_s,
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
