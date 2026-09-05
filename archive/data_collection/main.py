"""球关节数据采集系统 — CLI 入口。

用法:
    python -m data_collection.main run              # 默认配置采集
    python -m data_collection.main run --config config.json
    python -m data_collection.main run --static 3 --calib 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args)
    orch = Orchestrator(config)
    return orch.run()


def _load_config(args: argparse.Namespace) -> Config:
    if args.config:
        config = Config.from_json(Path(args.config))
    else:
        config = Config()

    # CLI 覆盖
    if args.mocap_ip:
        config.mocap.server_ip = args.mocap_ip
    if args.force_port:
        config.force.serial_port = args.force_port
    if args.static is not None:
        config.orchestrator.static_duration_s = args.static
    if args.calib is not None:
        config.orchestrator.calibration_duration_s = args.calib
    if args.duration is not None:
        config.orchestrator.exploration_duration_s = args.duration
    if args.no_mocap:
        config.output.enable_mocap_csv = False
    if args.no_imu:
        config.output.enable_imu_csv = False
    if args.no_force:
        config.output.enable_force_csv = False

    return config


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="球关节数据采集系统")
    subparsers = parser.add_subparsers(dest="command")

    # ---- run ----
    run_parser = subparsers.add_parser("run", help="启动采集会话")
    run_parser.add_argument("--config", "-c", help="JSON 配置文件路径")
    run_parser.add_argument("--mocap-ip", help="动捕服务器 IP")
    run_parser.add_argument("--force-port", help="力传感器串口")
    run_parser.add_argument("--static", type=float, help="静止段时长 (s)")
    run_parser.add_argument("--calib", type=float, help="标定段时长 (s)")
    run_parser.add_argument("--duration", "-d", type=float, help="探索段时长 (s)，默认无限")
    run_parser.add_argument("--no-mocap", action="store_true", help="禁用动捕")
    run_parser.add_argument("--no-imu", action="store_true", help="禁用 IMU")
    run_parser.add_argument("--no-force", action="store_true", help="禁用力量传感器")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
