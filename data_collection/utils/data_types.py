"""统一数据类型定义。"""

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


@dataclass
class Marker3D:
    x: float
    y: float
    z: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


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
        """一个帧可能产生多行 (每个 rigid body 一行)。"""
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
# IMU 数据
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

# IM948 数据包解析常量
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
    ax_with_g: float = 0.0
    ay_with_g: float = 0.0
    az_with_g: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    temperature_c: float = 0.0
    air_pressure_hpa: float = 0.0
    height_m: float = 0.0
    quat_w: float = 1.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0
    angle_x: float = 0.0
    angle_y: float = 0.0
    angle_z: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    steps: int = 0
    walking: int = 0
    running: int = 0
    biking: int = 0
    driving: int = 0
    nav_acc_x: float = 0.0
    nav_acc_y: float = 0.0
    nav_acc_z: float = 0.0
    adc_mv: float = 0.0
    gpio_raw: int = 0

    @classmethod
    def from_raw_packet(cls, data: bytes) -> Optional["ImuPacket"]:
        """从 IM948 BLE 原始数据包解析。"""
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
            packet.ax_with_g = _s16(data, i) * _SCALE_ACCEL
            packet.ay_with_g = _s16(data, i + 2) * _SCALE_ACCEL
            packet.az_with_g = _s16(data, i + 4) * _SCALE_ACCEL
            i += 6

        if tag & 0x0004:
            packet.gyro_x = _s16(data, i) * _SCALE_ANGLE_SPEED
            packet.gyro_y = _s16(data, i + 2) * _SCALE_ANGLE_SPEED
            packet.gyro_z = _s16(data, i + 4) * _SCALE_ANGLE_SPEED
            i += 6

        if tag & 0x0008:
            packet.mag_x = _s16(data, i) * _SCALE_MAG
            packet.mag_y = _s16(data, i + 2) * _SCALE_MAG
            packet.mag_z = _s16(data, i + 4) * _SCALE_MAG
            i += 6

        if tag & 0x0010:
            packet.temperature_c = _s16(data, i) * _SCALE_TEMPERATURE
            i += 2
            packet.air_pressure_hpa = _s24(data, i) * _SCALE_AIR_PRESSURE
            i += 3
            packet.height_m = _s24(data, i) * _SCALE_HEIGHT
            i += 3

        if tag & 0x0020:
            packet.quat_w = _s16(data, i) * _SCALE_QUAT
            packet.quat_x = _s16(data, i + 2) * _SCALE_QUAT
            packet.quat_y = _s16(data, i + 4) * _SCALE_QUAT
            packet.quat_z = _s16(data, i + 6) * _SCALE_QUAT
            i += 8

        if tag & 0x0040:
            packet.angle_x = _s16(data, i) * _SCALE_ANGLE
            packet.angle_y = _s16(data, i + 2) * _SCALE_ANGLE
            packet.angle_z = _s16(data, i + 4) * _SCALE_ANGLE
            i += 6

        if tag & 0x0080:
            packet.offset_x = _s16(data, i) / 1000.0
            packet.offset_y = _s16(data, i + 2) / 1000.0
            packet.offset_z = _s16(data, i + 4) / 1000.0
            i += 6

        if tag & 0x0100:
            packet.steps = _u32(data, i)
            i += 4
            activity = data[i]
            i += 1
            packet.walking = 1 if activity & 0x01 else 0
            packet.running = 1 if activity & 0x02 else 0
            packet.biking = 1 if activity & 0x04 else 0
            packet.driving = 1 if activity & 0x08 else 0

        if tag & 0x0200:
            packet.nav_acc_x = _s16(data, i) * _SCALE_ACCEL
            packet.nav_acc_y = _s16(data, i + 2) * _SCALE_ACCEL
            packet.nav_acc_z = _s16(data, i + 4) * _SCALE_ACCEL
            i += 6

        if tag & 0x0400:
            packet.adc_mv = _u16(data, i)
            i += 2

        if tag & 0x0800:
            packet.gpio_raw = data[i]

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
            "ax_with_g_mps2": self.ax_with_g,
            "ay_with_g_mps2": self.ay_with_g,
            "az_with_g_mps2": self.az_with_g,
            "gyro_x_dps": self.gyro_x,
            "gyro_y_dps": self.gyro_y,
            "gyro_z_dps": self.gyro_z,
            "mag_x": self.mag_x,
            "mag_y": self.mag_y,
            "mag_z": self.mag_z,
            "temperature_c": self.temperature_c,
            "air_pressure_hpa": self.air_pressure_hpa,
            "height_m": self.height_m,
            "quat_w": self.quat_w,
            "quat_x": self.quat_x,
            "quat_y": self.quat_y,
            "quat_z": self.quat_z,
            "angle_x_deg": self.angle_x,
            "angle_y_deg": self.angle_y,
            "angle_z_deg": self.angle_z,
            "offset_x_m": self.offset_x,
            "offset_y_m": self.offset_y,
            "offset_z_m": self.offset_z,
            "steps": self.steps,
            "walking": self.walking,
            "running": self.running,
            "biking": self.biking,
            "driving": self.driving,
            "nav_acc_x_mps2": self.nav_acc_x,
            "nav_acc_y_mps2": self.nav_acc_y,
            "nav_acc_z_mps2": self.nav_acc_z,
            "adc_mv": self.adc_mv,
            "gpio_raw": self.gpio_raw,
        }


# ---------------------------------------------------------------------------
# 力传感器数据
# ---------------------------------------------------------------------------

FORCE_CSV_COLUMNS = [
    "pc_timestamp_ns",
    "pc_receive_unix_time_ms",
    "ch1", "ch2", "ch3", "ch4",
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

    def __iter__(self):
        return iter((self.ch1, self.ch2, self.ch3, self.ch4, self.ch5, self.ch6))

    def to_csv_row(self) -> dict:
        return {
            "pc_timestamp_ns": self.pc_timestamp_ns,
            "pc_receive_unix_time_ms": self.pc_receive_unix_time_ms,
            "ch1": self.ch1,
            "ch2": self.ch2,
            "ch3": self.ch3,
            "ch4": self.ch4,
            "ch5": self.ch5,
            "ch6": self.ch6,
        }


# ---------------------------------------------------------------------------
# 舵机状态 (开环，无编码器反馈)
# ---------------------------------------------------------------------------

SERVO_CSV_COLUMNS = [
    "pc_timestamp_ns",
    "pc_receive_unix_time_ms",
    "servo_1_target_deg", "servo_2_target_deg", "servo_3_target_deg", "servo_4_target_deg",
    "target_pitch", "target_yaw",
    "current_pitch", "current_yaw",
    "segment_id",
    "phase",
]


@dataclass
class ServoState:
    pc_timestamp_ns: int
    pc_receive_unix_time_ms: int
    target_angles: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    target_pitch: float = 0.0
    target_yaw: float = 0.0
    current_pitch: float = 0.0
    current_yaw: float = 0.0
    segment_id: int = -1

    def to_csv_row(self) -> dict:
        row = {
            "pc_timestamp_ns": self.pc_timestamp_ns,
            "pc_receive_unix_time_ms": self.pc_receive_unix_time_ms,
        }
        for i in range(4):
            row[f"servo_{i+1}_target_deg"] = (
                self.target_angles[i] if i < len(self.target_angles) else ""
            )
        row["target_pitch"] = self.target_pitch
        row["target_yaw"] = self.target_yaw
        row["current_pitch"] = self.current_pitch
        row["current_yaw"] = self.current_yaw
        row["segment_id"] = self.segment_id
        return row


# ---------------------------------------------------------------------------
# 会话级 CSV 列定义 (含 phase)
# ---------------------------------------------------------------------------

MOCAP_CSV_COLUMNS = [
    "pc_timestamp_ns", "pc_receive_unix_time_ms",
    "frame_index", "hardware_timestamp",
    "rigid_body_id", "rigid_body_x", "rigid_body_y", "rigid_body_z",
    "rigid_body_qx", "rigid_body_qy", "rigid_body_qz", "rigid_body_qw",
    "phase",
]
