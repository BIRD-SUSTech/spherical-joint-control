"""集中配置管理。所有参数通过 dataclass 定义，支持 JSON 文件加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MocapConfig:
    server_ip: str = "10.1.1.198"
    data_cache_size: int = 100


@dataclass
class ImuConfig:
    device_address: str = "A5:B2:90:FF:4A:12"
    device_name_keyword: str = "im948"
    report_hz: int = 100
    report_tag: int = 0x0025
    notify_characteristic: int = 0x0007
    write_characteristic: int = 0x0005
    scan_timeout_s: float = 10.0
    is_compass_on: int = 0
    barometer_filter: int = 2


@dataclass
class ForceConfig:
    serial_port: str = "COM6"
    baudrate: int = 19200
    data_bits: int = 8
    stop_bits: int = 1
    parity: str = "N"
    timeout: float = 0.5
    rs485_mode: bool = False
    rs485_tx_pin: str = "RTS"
    rs485_tx_level: bool = True
    slave_address: int = 0x01
    reg_start_address: int = 0x000B
    channel_count: int = 4
    scale_factor: float = 1.0
    sample_interval_ms: int = 100


@dataclass
class ServoConfig:
    enabled: bool = False
    port: str = "COM5"
    baudrate: int = 115200
    servo_count: int = 4
    # 轨迹配置
    trajectory_type: str = "idle"      # "idle" | "waypoints" | "sine" | "circle"
    trajectory_file: Optional[str] = None
    trajectory_waypoints: list = field(default_factory=list)  # [(pitch, yaw, dur_s), ...]
    trajectory_amplitude_deg: float = 10.0   # circle/sine 振幅
    trajectory_period_s: float = 10.0        # circle/sine 周期
    trajectory_steps: int = 100              # circle/sine 路径点数
    # 标定
    calibration_amplitude_deg: float = 20.0  # 标定扫频振幅
    # 预紧
    pretension_mm: float = 2.0               # 线缆预紧量，正值=缩短，防松弛脱盘
    # PID 反馈
    mocap_filter_tau_s: float = 0.0          # 动捕反馈 EMA 滤波时间常数 (s)，0=关闭
    rigid_body_id: int = 0             # 用于姿态反馈的 mocap 刚体 ID
    servo_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])


@dataclass
class OutputConfig:
    root_dir: Path = Path("data_collection/sessions")
    session_prefix: str = "session"
    enable_mocap_csv: bool = True
    enable_imu_csv: bool = True
    enable_force_csv: bool = True
    enable_servo_csv: bool = True
    queue_maxsize: int = 5000
    flush_interval_rows: int = 10


@dataclass
class OrchestratorConfig:
    static_duration_s: float = 3.0
    calibration_duration_s: float = 15.0
    exploration_duration_s: Optional[float] = None  # None = unlimited
    startup_timeout_s: float = 30.0
    shutdown_timeout_s: float = 5.0
    drain_timeout_s: float = 3.0
    progress_report_interval_s: float = 5.0


@dataclass
class Config:
    mocap: MocapConfig = field(default_factory=MocapConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    force: ForceConfig = field(default_factory=ForceConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            mocap=MocapConfig(**data.get("mocap", {})),
            imu=ImuConfig(**data.get("imu", {})),
            force=ForceConfig(**data.get("force", {})),
            servo=ServoConfig(**data.get("servo", {})),
            output=_make_output_config(data.get("output", {})),
            orchestrator=OrchestratorConfig(**data.get("orchestrator", {})),
        )

    def to_json(self, path: Path) -> None:
        data = {
            "mocap": self.mocap.__dict__,
            "imu": self.imu.__dict__,
            "force": self.force.__dict__,
            "servo": self.servo.__dict__,
            "output": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.output.__dict__.items()
            },
            "orchestrator": self.orchestrator.__dict__,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _make_output_config(data: dict) -> OutputConfig:
    if "root_dir" in data:
        data = {**data, "root_dir": Path(data["root_dir"])}
    return OutputConfig(**data)
