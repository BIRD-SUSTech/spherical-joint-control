"""会话目录管理（复用 archive/data_collection/utils/session.py 设计）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SessionPaths:
    dir: Path
    mocap_csv: Path
    imu_csv: Path
    force_csv: Path
    servo_csv: Path
    metadata_json: Path


class SessionManager:
    """创建时间戳会话目录，同名自动递增后缀。"""

    def __init__(self, root_dir: Path, prefix: str = "session"):
        self._root = Path(root_dir)
        self._prefix = prefix

    def create_session(self) -> SessionPaths:
        session_dir = self._make_unique_dir()
        return SessionPaths(
            dir=session_dir,
            mocap_csv=session_dir / "mocap_data.csv",
            imu_csv=session_dir / "imu_data.csv",
            force_csv=session_dir / "force_data.csv",
            servo_csv=session_dir / "servo_data.csv",
            metadata_json=session_dir / "session_metadata.json",
        )

    def _make_unique_dir(self) -> Path:
        base_name = datetime.now().strftime(f"{self._prefix}_%Y%m%d_%H%M%S")
        session_dir = self._root / base_name
        if not session_dir.exists():
            session_dir.mkdir(parents=True, exist_ok=True)
            return session_dir
        index = 2
        while True:
            candidate = self._root / f"{base_name}_{index:02d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            index += 1
