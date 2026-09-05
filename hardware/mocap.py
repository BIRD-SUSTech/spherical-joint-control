"""动捕读帧：Nokov SDK 后台线程轮询最后帧，取欧拉角。

从 archive/example_code/steering_motor/Scripts/CloseLoop/motion_capture.py 提炼。
真实 SDK 在 connect() 时惰性导入，因此 --mock 模式无需安装厂商 SDK。

姿态命名：本模块只返回动捕原始欧拉角 (pitch, roll, yaw)，不绑定关节语义；
欧拉角 → 前后/左右 的映射在 control 层用标定配置决定（设计文档 §7.3）。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """动捕解算的欧拉角（度）。"""
    pitch: float
    roll: float
    yaw: float
    frame_index: int


class MocapReader:
    """Nokov 动捕：后台线程轮询最后帧，get_pose() 返回最新欧拉角。"""

    def __init__(self, server_ip: str = "10.1.1.198", rigid_body_index: int = 0):
        self.server_ip = server_ip
        self.rigid_body_index = rigid_body_index
        self._sdk = None
        self._client = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._pose: Pose | None = None

    def connect(self) -> bool:
        from nokov import nokovsdk  # 惰性导入，mock 无需 SDK

        try:
            self._sdk = nokovsdk
            self._client = nokovsdk.PySDKClient()
            ret = self._client.Initialize(bytes(self.server_ip, encoding="utf8"))
            if ret != 0:
                logger.error("连接 Nokov 失败: [%s]", ret)
                return False
            self._client.PySetVerbosityLevel(0)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Nokov 初始化异常: %s", e)
            return False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            frame = self._client.PyGetLastFrameOfMocapData()
            if frame:
                try:
                    self._process(frame)
                except Exception:  # noqa: BLE001
                    logger.exception("动捕帧处理异常")
                finally:
                    self._client.PyNokovFreeFrame(frame)

    def _process(self, frame) -> None:
        data = frame.contents
        if data.nRigidBodies <= self.rigid_body_index:
            return

        pitch = roll = yaw = 0.0
        ext = data.FrameExtendData
        for i in range(ext.nExtendDataNum):
            if ext.extendData[i].type == self._sdk.ExtendDataType.ExtendDataRigidBody.value:
                rb_ext = ext.extendData[i]
                if rb_ext.number > self.rigid_body_index:
                    e = rb_ext.ExtendDataUnion.RigidBodyExtendData[self.rigid_body_index]
                    pitch, roll, yaw = e.pitch, e.roll, e.yaw
                break

        with self._lock:
            self._pose = Pose(pitch=pitch, roll=roll, yaw=yaw, frame_index=data.iFrame)

    def get_pose(self) -> Pose | None:
        with self._lock:
            return self._pose


class MockMocap:
    """离线自检用：返回静态零姿态（无硬件、无 SDK）。"""

    def __init__(self):
        self._running = False
        self._frame = 0

    def connect(self) -> bool:
        return True

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def get_pose(self) -> Pose | None:
        if not self._running:
            return None
        self._frame += 1
        return Pose(pitch=0.0, roll=0.0, yaw=0.0, frame_index=self._frame)
