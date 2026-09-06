"""动捕读帧：Nokov SDK 后台线程轮询最后帧，取欧拉角 + 四元数/位姿。

从 archive/example_code/steering_motor/Scripts/CloseLoop/motion_capture.py 提炼。
真实 SDK 在 connect() 时惰性导入，因此 --mock 模式无需安装厂商 SDK。

双重用途（单一事实源）：
    - 控制层（L3）：get_pose() 轮询最新欧拉角（pitch/roll/yaw）。
    - 采集层（L1）：on_frame 回调订阅完整帧（含刚体四元数/位姿）用于落盘。

姿态命名：本模块返回动捕原始欧拉角 (pitch, roll, yaw)，不绑定关节语义；
欧拉角 → 前后/左右 的映射在 control 层用标定配置决定（设计文档 §7.3）。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """动捕解算的一帧刚体姿态。

    pitch/roll/yaw：SDK 扩展欧拉角（度）。
    x/y/z：刚体位置；qx/qy/qz/qw：刚体姿态四元数（采集层落盘用）。
    """
    pitch: float
    roll: float
    yaw: float
    frame_index: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


class MocapReader:
    """Nokov 动捕：后台线程轮询最后帧；get_pose() 轮询，on_frame 订阅新帧。"""

    def __init__(
        self,
        server_ip: str = "10.1.1.198",
        rigid_body_index: int = 0,
        on_frame: Optional[Callable[[Pose], None]] = None,
    ):
        self.server_ip = server_ip
        self.rigid_body_index = rigid_body_index
        self._on_frame = on_frame
        self._sdk = None
        self._client = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._pose: Optional[Pose] = None

    def connect(self) -> bool:
        try:
            from nokov import nokovsdk  # 惰性导入，mock 无需 SDK

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

    def join(self, timeout: float | None = None) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

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

        rb = data.RigidBodies[self.rigid_body_index]
        pitch = roll = yaw = 0.0
        ext = data.FrameExtendData
        for i in range(ext.nExtendDataNum):
            if ext.extendData[i].type == self._sdk.ExtendDataType.ExtendDataRigidBody.value:
                rb_ext = ext.extendData[i]
                if rb_ext.number > self.rigid_body_index:
                    e = rb_ext.ExtendDataUnion.RigidBodyExtendData[self.rigid_body_index]
                    pitch, roll, yaw = e.pitch, e.roll, e.yaw
                break

        pose = Pose(
            pitch=pitch, roll=roll, yaw=yaw, frame_index=data.iFrame,
            x=rb.x, y=rb.y, z=rb.z,
            qx=rb.qx, qy=rb.qy, qz=rb.qz, qw=rb.qw,
        )
        with self._lock:
            self._pose = pose
        if self._on_frame is not None:
            try:
                self._on_frame(pose)
            except Exception:  # noqa: BLE001
                logger.exception("on_frame 回调异常")

    def get_pose(self) -> Optional[Pose]:
        with self._lock:
            return self._pose


class MockMocap:
    """离线自检用：返回静态零姿态（无硬件、无 SDK），on_frame 同步触发。"""

    def __init__(self, on_frame: Optional[Callable[[Pose], None]] = None):
        self._on_frame = on_frame
        self._running = False
        self._frame = 0

    def connect(self) -> bool:
        return True

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def join(self, timeout: float | None = None) -> None:
        pass

    def get_pose(self) -> Optional[Pose]:
        if not self._running:
            return None
        self._frame += 1
        pose = Pose(pitch=0.0, roll=0.0, yaw=0.0, frame_index=self._frame,
                    qw=1.0)
        if self._on_frame is not None:
            self._on_frame(pose)
        return pose
