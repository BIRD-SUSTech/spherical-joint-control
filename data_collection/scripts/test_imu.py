"""IM948 BLE IMU 独立连接测试。

扫描并连接 IM948，等待数据包，报告采样率和姿态数据。

用法:
    python scripts/test_imu.py [--timeout 20] [--samples 10]

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

from data_collection.config import ImuConfig  # noqa: E402
from data_collection.sensor_collectors.imu_collector import ImuCollector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="IM948 BLE IMU 连接测试")
    parser.add_argument("--timeout", type=float, default=20.0, help="总等待时长 (s)，含 BLE 扫描")
    parser.add_argument("--samples", type=int, default=10, help="确认数据流所需的包数")
    args = parser.parse_args()

    q: queue.Queue = queue.Queue(maxsize=2000)
    start_ev, stop_ev = threading.Event(), threading.Event()
    collector = ImuCollector(ImuConfig(), q, start_ev, stop_ev)

    collector.start()
    print(f"[INFO] 正在扫描/连接 IM948，最长等待 {args.timeout:.0f}s...")

    deadline = time.perf_counter() + args.timeout
    while time.perf_counter() < deadline and q.qsize() < args.samples:
        time.sleep(0.5)

    collector.stop()
    collector.join(timeout=5.0)

    count = q.qsize()
    if count == 0:
        print("[FAIL] 未收到任何 IMU 数据包（检查 BLE 地址/设备是否开机）")
        return 1

    p = q.get()
    print(f"[INFO] 共收到 {count} 包")
    print(f"[INFO] subscribe_tag={p.subscribe_tag} "
          f"quat=({p.quat_w:.3f},{p.quat_x:.3f},{p.quat_y:.3f},{p.quat_z:.3f}) "
          f"angle=({p.angle_x:.1f},{p.angle_y:.1f},{p.angle_z:.1f})")
    print(f"[PASS] IMU 测试通过 (约 {count / (args.timeout if count else 1):.0f} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
