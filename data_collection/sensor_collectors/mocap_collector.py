"""Nokov 运动捕捉采集线程。

从 imu948_and_mocap_logger.py 重构：回调模式改为 queue.put。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from typing import Optional

from nokov import nokovsdk

from ..config import MocapConfig
from ..utils.data_types import Marker3D, MocapFrame, RigidBody

logger = logging.getLogger(__name__)


def _py_msg_func(level: int, message):
    text = message.decode("utf-8", errors="replace")
    if level <= 1:
        logger.error("Nokov SDK: %s", text)
    elif level == 2:
        logger.warning("Nokov SDK: %s", text)
    elif level == 3:
        logger.info("Nokov SDK: %s", text)
    else:
        logger.debug("Nokov SDK: %s", text)


class MocapCollector:
    """Nokov 动捕系统采集器。内部线程轮询 SDK 帧数据。"""

    def __init__(
        self,
        config: MocapConfig,
        output_queue: queue.Queue,
        start_event: threading.Event,
        stop_event: threading.Event,
    ):
        self._config = config
        self._output_queue = output_queue
        self._start_event = start_event
        self._stop_event = stop_event
        self._client = nokovsdk.PySDKClient()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._dropped_count = 0

    def connect(self) -> bool:
        ret = self._client.Initialize(
            bytes(self._config.server_ip, encoding="utf8")
        )
        if ret == 0:
            logger.info("Connected to Nokov server %s", self._config.server_ip)
            return True
        logger.error("Nokov connection failed (ret=%d)", ret)
        return False

    def start(self) -> None:
        self._client.PySetVerbosityLevel(0)
        self._client.PySetMessageCallback(_py_msg_func)
        self._thread = threading.Thread(
            target=self._loop, name="MocapCollector", daemon=True
        )
        self._thread.start()
        logger.info("Mocap collector thread started")

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        last_frame_index = -1

        while not self._stop_event.is_set():
            frame_ptr = self._client.PyGetLastFrameOfMocapData()
            if not frame_ptr:
                time.sleep(0.001)
                continue

            try:
                frame_data = frame_ptr.contents
                if frame_data.iFrame == last_frame_index:
                    continue
                last_frame_index = frame_data.iFrame

                mocap_frame = self._extract(frame_data)
                if mocap_frame is not None:
                    try:
                        self._output_queue.put_nowait(mocap_frame)
                    except queue.Full:
                        with self._lock:
                            self._dropped_count += 1
            except Exception:
                logger.error("Mocap thread error:\n%s", traceback.format_exc())
            finally:
                self._client.PyNokovFreeFrame(frame_ptr)

    def _extract(self, frame_data) -> Optional[MocapFrame]:
        if not frame_data:
            return None

        now_ns = time.perf_counter_ns()
        now_ms = int(time.time() * 1000)

        markersets = {}
        for i_ms in range(frame_data.nMarkerSets):
            ms = frame_data.MocapData[i_ms]
            name = ms.szName.decode("utf-8", errors="replace")
            markers = [Marker3D(m[0], m[1], m[2]) for m in ms.Markers[: ms.nMarkers]]
            markersets[name] = markers

        rigid_bodies = []
        for i_rb in range(frame_data.nRigidBodies):
            rb = frame_data.RigidBodies[i_rb]
            markers = [
                Marker3D(rb.Markers[j][0], rb.Markers[j][1], rb.Markers[j][2])
                for j in range(rb.nMarkers)
            ]
            rigid_bodies.append(
                RigidBody(
                    id=rb.ID,
                    x=rb.x, y=rb.y, z=rb.z,
                    qx=rb.qx, qy=rb.qy, qz=rb.qz, qw=rb.qw,
                    markers=markers,
                )
            )

        return MocapFrame(
            pc_timestamp_ns=now_ns,
            pc_receive_unix_time_ms=now_ms,
            frame_index=frame_data.iFrame,
            hardware_timestamp=frame_data.iTimeStamp,
            markersets=markersets,
            rigid_bodies=rigid_bodies,
        )
