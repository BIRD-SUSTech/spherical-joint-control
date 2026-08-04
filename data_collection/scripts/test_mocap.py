"""Nokov 动捕独立连接测试。

连接动捕服务器，采集若干秒数据，报告帧率和可见刚体。

用法:
    python scripts/test_mocap.py [--duration 5] [--ip 10.1.1.198]

退出码: 0=通过, 1=失败.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_collection.config import Config, MocapConfig  # noqa: E402
from data_collection.sensor_collectors.mocap_collector import MocapCollector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Nokov 动捕连接测试")
    parser.add_argument("--duration", type=float, default=5.0, help="采集时长 (s)")
    parser.add_argument("--ip", type=str, default=None, help="动捕服务器 IP")
    args = parser.parse_args()

    mocap_cfg = MocapConfig()
    if args.ip:
        mocap_cfg.server_ip = args.ip

    q: queue.Queue = queue.Queue(maxsize=1000)
    start_ev, stop_ev = threading.Event(), threading.Event()
    collector = MocapCollector(mocap_cfg, q, start_ev, stop_ev)

    if not collector.connect():
        print(f"[FAIL] 无法连接动捕服务器 {mocap_cfg.server_ip} (请确认 Nokov 软件已启动)")
        return 1

    collector.start()
    print(f"[OK] 已连接 {mocap_cfg.server_ip}，采集 {args.duration}s...")

    time.sleep(args.duration)
    collector.stop()
    collector.join(timeout=3.0)

    count = q.qsize()
    print(f"[INFO] 收到 {count} 帧")

    if count == 0:
        print("[FAIL] 未收到任何动捕帧")
        return 1

    frame = q.get()
    print(f"[INFO] 示例帧 #{frame.frame_index}, "
          f"刚体: {[(rb.id, f'({rb.x:.1f},{rb.y:.1f},{rb.z:.1f})') for rb in frame.rigid_bodies]}")
    print(f"[OK] 帧率约 {count / args.duration:.0f} Hz")
    print("[PASS] 动捕测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
