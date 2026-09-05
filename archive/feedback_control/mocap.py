"""Nokov 动捕读取（最小实现，只依赖动捕，不触碰 IMU/力传感器）。

反馈量：刚体欧拉角 pitch（左右，DOF1）/ yaw（前后，DOF2）。
优先直接取 SDK 刚体扩展数据（RigidBodyExtendData，零点在 Nokov Seeker 中设置，
与参考 CloseLoop 一致）；扩展数据缺失时退化为“首帧四元数为参考 + 四元数分解”。

命名约定（沿用当前系统的 pitch / yaw）：
    pitch = DOF1 = 绕 X 轴倾斜 = 舵机对 1↔3
    yaw   = DOF2 = 绕 Y 轴倾斜 = 舵机对 2↔4
实机验证（--dry-run 手动把球杆偏向 1 号舵机，几何定义应为 pitch<0、yaw≈0）：
    SDK 扩展数据实测 pitch≈0、roll<0，说明 SDK 的 roll/pitch 与本项目
    pitch/yaw 正好互换，读取时必须交换（pitch←roll, yaw←pitch）。
    （SDK 的 yaw 是绕 Z 轴自转，绳驱不可控、不使用。）
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

from nokov import nokovsdk

from .quat import quat_to_pitch_yaw

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    pitch: float = 0.0   # DOF1（左右，绕 X，对应舵机对 1/3）
    yaw: float = 0.0     # DOF2（前后，绕 Y，对应舵机对 2/4）
    frame_index: int = -1


class MocapReader:
    def __init__(self, server_ip: str = "10.1.1.198", rigid_body_index: int = 0):
        self.server_ip = server_ip
        self.rigid_body_index = rigid_body_index
        self._client = nokovsdk.PySDKClient()
        self._lock = threading.Lock()
        self._pose = Pose()
        self._ref_q = None
        self._thread = None
        self._running = False

    def connect(self) -> bool:
        ret = self._client.Initialize(bytes(self.server_ip, encoding="utf8"))
        if ret == 0:
            logger.info("mocap 已连接 %s", self.server_ip)
            return True
        logger.error("mocap 连接失败 ret=%d", ret)
        return False

    def start(self) -> None:
        self._client.PySetVerbosityLevel(0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="MocapReader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_pose(self) -> Pose:
        with self._lock:
            return self._pose

    # ---- 内部 ----

    def _loop(self) -> None:
        last_frame = -1
        while self._running:
            ptr = self._client.PyGetLastFrameOfMocapData()
            if not ptr:
                time.sleep(0.001)
                continue
            try:
                frame = ptr.contents
                if frame.iFrame == last_frame:
                    continue
                last_frame = frame.iFrame
                with self._lock:
                    self._pose = self._extract(frame)
            except Exception:
                logger.exception("mocap 解析出错")
            finally:
                self._client.PyNokovFreeFrame(ptr)

    def _extract(self, frame) -> Pose:
        pitch = yaw = 0.0
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        # 1) 取刚体四元数（用于兜底）
        if frame.nRigidBodies > self.rigid_body_index:
            rb = frame.RigidBodies[self.rigid_body_index]
            qx, qy, qz, qw = rb.qx, rb.qy, rb.qz, rb.qw

        # 2) 优先取 SDK 扩展欧拉角（交换：pitch←roll, yaw←pitch，见模块 docstring）
        got_euler = False
        try:
            fext = frame.FrameExtendData
            for i in range(fext.nExtendDataNum):
                if fext.extendData[i].type == nokovsdk.ExtendDataType.ExtendDataRigidBody.value:
                    ext = fext.extendData[i]
                    if self.rigid_body_index < ext.number:
                        rb_ext = ext.ExtendDataUnion.RigidBodyExtendData[self.rigid_body_index]
                        pitch, yaw = rb_ext.roll, rb_ext.pitch
                        got_euler = True
                    break
        except Exception:
            pass

        # 3) 兜底：四元数分解（首帧为参考）
        if not got_euler:
            if self._ref_q is None:
                self._ref_q = (qw, qx, qy, qz)
            pitch, yaw = quat_to_pitch_yaw(qw, qx, qy, qz, self._ref_q)

        return Pose(pitch=float(pitch), yaw=float(yaw), frame_index=frame.iFrame)


class MockMocap:
    """无硬件测试桩：返回缓慢正弦姿态，用于离线跑通整条链路。"""

    def __init__(self, amp: float = 5.0, period: float = 6.0):
        self._amp = amp
        self._period = period
        self._t0 = time.perf_counter()

    def connect(self) -> bool:
        return True

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        pass

    def get_pose(self) -> Pose:
        t = time.perf_counter() - self._t0
        return Pose(
            pitch=self._amp * math.sin(2 * math.pi * t / self._period),
            yaw=self._amp * math.cos(2 * math.pi * t / self._period),
            frame_index=int(t * 100),
        )
