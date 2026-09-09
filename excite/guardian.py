"""安全监护：关节角超限回中位（M4，L2 激励层）。

独立线程监控动捕实测前后/左右角，超过 limit_deg（70°）即触发急停。
急停 = 回中位 send_pair(0,0)，不放线（id=0 会杆垂落，M1 实机反馈）。
"""

from __future__ import annotations

import logging
import threading
import time

from control.calibration import Calibration

logger = logging.getLogger(__name__)

LIMIT_DEG = 70.0
POLL_S = 0.005  # 监控周期（200Hz，受动捕帧率 ~90Hz 上限）


class Guardian:
    def __init__(self, mocap, bus, calibration: Calibration | None = None,
                 limit_deg: float = LIMIT_DEG):
        self._mocap = mocap
        self._bus = bus
        self._calib = calibration or Calibration.default()
        self._limit_deg = limit_deg
        self._triggered = False
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="Guardian", daemon=True)
        self._thread.start()
        logger.info("Guardian 启动（限位 %.0f°，急停=回中位）", self._limit_deg)

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            pose = self._mocap.get_pose()
            if pose is not None:
                fb, lr = self._calib.map_pose(pose.roll, pose.pitch)
                if abs(fb) > self._limit_deg or abs(lr) > self._limit_deg:
                    logger.error("关节角超限 fb=%.1f° lr=%.1f°（限位 %.0f°），急停回中位",
                                 fb, lr, self._limit_deg)
                    self._bus.send_pair_tension(0, 0, 0, 0)  # 差分+共模同时回零
                    self._triggered = True
                    break
            time.sleep(POLL_S)
