"""统一采集 schema：CSV 列定义 + 数据类型（唯一事实源，设计文档 §6）。

物理命名：前后(front_back, id=1) / 左右(left_right, id=2)。
舵机指令单位：PWM offset（int16，2 路差分，固件内耦合）。
segment 语义：segment_id >= 0 为数据段（供建模按段切分），< 0 为非采集段（dwell/标定）。
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CollectionPhase(Enum):
    STATIC = "static"
    CALIBRATION = "calibration"
    EXPLORATION = "exploration"


# ---------------------------------------------------------------------------
# 动捕数据
# ---------------------------------------------------------------------------

MOCAP_CSV_COLUMNS = [
    "pc_timestamp_ns", "pc_receive_unix_time_ms",
    "frame_index", "hardware_timestamp",
    "rigid_body_id", "rigid_body_x", "rigid_body_y", "rigid_body_z",
    "rigid_body_qx", "rigid_body_qy", "rigid_body_qz", "rigid_body_qw",
    "phase",
]


@dataclass
class Marker3D:
    x: float
    y: float
    z: float


@dataclass
class RigidBody:
    id: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    markers: List[Marker3D] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rigid_body_id": self.id,
            "rigid_body_x": self.x,
            "rigid_body_y": self.y,
            "rigid_body_z": self.z,
            "rigid_body_qx": self.qx,
            "rigid_body_qy": self.qy,
            "rigid_body_qz": self.qz,
            "rigid_body_qw": self.qw,
        }


@dataclass
class MocapFrame:
    pc_timestamp_ns: int
    pc_receive_unix_time_ms: int
    frame_index: int
    hardware_timestamp: int
    markersets: Dict[str, List[Marker3D]] = field(default_factory=dict)
    rigid_bodies: List[RigidBody] = field(default_factory=list)

    def to_csv_rows(self) -> list[dict]:
        base = {
            "pc_timestamp_ns": self.pc_timestamp_ns,
            "pc_receive_unix_time_ms": self.pc_receive_unix_time_ms,
            "frame_index": self.frame_index,
            "hardware_timestamp": self.hardware_timestamp,
        }
        if not self.rigid_bodies:
            row = {**base}
            for key in ("rigid_body_id", "rigid_body_x", "rigid_body_y", "rigid_body_z",
                        "rigid_body_qx", "rigid_body_qy", "rigid_body_qz", "rigid_body_qw"):
                row[key] = ""
            return [row]
        return [{**base, **rb.to_dict()} for rb in self.rigid_bodies]


# ---------------------------------------------------------------------------
# IMU 数据（IM948 协议解析复用 archive/data_collection）
# ---------------------------------------------------------------------------

IMU_CSV_COLUMNS = [
    "pc_timestamp_ns",
    "pc_receive_unix_time_ms",
    "chip_time_ms",
    "subscribe_tag",
    "ax_no_g_mps2", "ay_no_g_mps2", "az_no_g_mps2",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "phase",
]

_SCALE_ACCEL = 0.00478515625
_SCALE_QUAT = 0.000030517578125
_SCALE_ANGLE = 0.0054931640625
_SCALE_ANGLE_SPEED = 0.06103515625
_SCALE_MAG = 0.15106201171875
_SCALE_TEMPERATURE = 0.01
_SCALE_AIR_PRESSURE = 0.0002384185791
_SCALE_HEIGHT = 0.0010728836


def _s16(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<h", buf, offset)[0]


def _u16(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<H", buf, offset)[0]


def _u32(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def _s24(buf: bytes, offset: int) -> int:
    value = buf[offset] | (buf[offset + 1] << 8) | (buf[offset + 2] << 16)
    if value & 0x800000:
        value -= 0x1000000
    return value


@dataclass
class ImuPacket:
    pc_timestamp_ns: int = 0
    pc_receive_unix_time_ms: int = 0
    chip_time_ms: int = 0
    subscribe_tag: str = ""
    ax_no_g: float = 0.0
    ay_no_g: float = 0.0
    az_no_g: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    quat_w: float = 1.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0

    @classmethod
    def from_raw_packet(cls, data: bytes) -> Optional["ImuPacket"]:
        """从 IM948 BLE 原始数据包解析（订阅标签驱动）。"""
        if len(data) < 7 or data[0] != 0x11:
            return None

        now_ns = time.perf_counter_ns()
        now_ms = int(time.time() * 1000)
        tag = (data[2] << 8) | data[1]
        chip_time_ms = _u32(data, 3)
        packet = cls(
            pc_timestamp_ns=now_ns,
            pc_receive_unix_time_ms=now_ms,
            chip_time_ms=chip_time_ms,
            subscribe_tag=f"0x{tag:04X}",
        )

        i = 7
        if tag & 0x0001:
            packet.ax_no_g = _s16(data, i) * _SCALE_ACCEL
            packet.ay_no_g = _s16(data, i + 2) * _SCALE_ACCEL
            packet.az_no_g = _s16(data, i + 4) * _SCALE_ACCEL
            i += 6
        if tag & 0x0002:
            i += 6  # ax_with_g / ay_with_g / az_with_g（本 schema 不落盘）
        if tag & 0x0004:
            packet.gyro_x = _s16(data, i) * _SCALE_ANGLE_SPEED
            packet.gyro_y = _s16(data, i + 2) * _SCALE_ANGLE_SPEED
            packet.gyro_z = _s16(data, i + 4) * _SCALE_ANGLE_SPEED
            i += 6
        if tag & 0x0008:
            i += 6  # mag
        if tag & 0x0010:
            i += 2 + 3 + 3  # temperature + pressure + height
        if tag & 0x0020:
            packet.quat_w = _s16(data, i) * _SCALE_QUAT
            packet.quat_x = _s16(data, i + 2) * _SCALE_QUAT
            packet.quat_y = _s16(data, i + 4) * _SCALE_QUAT
            packet.quat_z = _s16(data, i + 6) * _SCALE_QUAT
            i += 8
        if tag & 0x0040:
            i += 6  # angle
        if tag & 0x0080:
            i += 6  # offset
        if tag & 0x0100:
            i += 5  # steps + activity
        if tag & 0x0200:
            i += 6  # nav acc
        if tag & 0x0400:
            i += 2  # adc
        return packet

    def to_csv_row(self) -> dict:
        return {
            "pc_timestamp_ns": self.pc_timestamp_ns,
            "pc_receive_unix_time_ms": self.pc_receive_unix_time_ms,
            "chip_time_ms": self.chip_time_ms,
            "subscribe_tag": self.subscribe_tag,
            "ax_no_g_mps2": self.ax_no_g,
            "ay_no_g_mps2": self.ay_no_g,
            "az_no_g_mps2": self.az_no_g,
            "gyro_x_dps": self.gyro_x,
            "gyro_y_dps": self.gyro_y,
            "gyro_z_dps": self.gyro_z,
            "quat_w": self.quat_w,
            "quat_x": self.quat_x,
            "quat_y": self.quat_y,
            "quat_z": self.quat_z,
        }


# ---------------------------------------------------------------------------
# 力传感器数据（六维力，MODBUS-RTU）
# ---------------------------------------------------------------------------

FORCE_CSV_COLUMNS = [
    "pc_timestamp_ns",
    "pc_receive_unix_time_ms",
    "ch1", "ch2", "ch3", "ch4", "ch5", "ch6",
    "phase",
]


@dataclass
class ForceData:
    pc_timestamp_ns: int
    pc_receive_unix_time_ms: int
    ch1: float = 0.0
    ch2: float = 0.0
    ch3: float = 0.0
    ch4: float = 0.0
    ch5: float = 0.0
    ch6: float = 0.0

    def to_csv_row(self) -> dict:
        return {
            "pc_timestamp_ns": self.pc_timestamp_ns,
            "pc_receive_unix_time_ms": self.pc_receive_unix_time_ms,
            "ch1": self.ch1, "ch2": self.ch2, "ch3": self.ch3,
            "ch4": self.ch4, "ch5": self.ch5, "ch6": self.ch6,
        }


# ---------------------------------------------------------------------------
# 舵机状态（闭环，offset 空间）
# ---------------------------------------------------------------------------

SERVO_CSV_COLUMNS = [
    "pc_timestamp_ns",
    "pc_receive_unix_time_ms",
    "t_s",
    "target_front_back_deg", "target_left_right_deg",
    "current_front_back_deg", "current_left_right_deg",
    "servo_front_back_offset", "servo_left_right_offset",
    "segment_id",
    "phase",
]


@dataclass
class ServoState:
    pc_timestamp_ns: int
    pc_receive_unix_time_ms: int
    t_s: float = 0.0
    target_front_back_deg: float = 0.0
    target_left_right_deg: float = 0.0
    current_front_back_deg: float = 0.0
    current_left_right_deg: float = 0.0
    servo_front_back_offset: int = 0
    servo_left_right_offset: int = 0
    segment_id: int = -1

    def to_csv_row(self) -> dict:
        return {
            "pc_timestamp_ns": self.pc_timestamp_ns,
            "pc_receive_unix_time_ms": self.pc_receive_unix_time_ms,
            "t_s": self.t_s,
            "target_front_back_deg": self.target_front_back_deg,
            "target_left_right_deg": self.target_left_right_deg,
            "current_front_back_deg": self.current_front_back_deg,
            "current_left_right_deg": self.current_left_right_deg,
            "servo_front_back_offset": self.servo_front_back_offset,
            "servo_left_right_offset": self.servo_left_right_offset,
            "segment_id": self.segment_id,
        }
