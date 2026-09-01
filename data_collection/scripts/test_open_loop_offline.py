"""开环激励采集离线自测（无需任何硬件 / SDK）。

验证：open_loop 模式下，控制器生成大摆幅平滑激励、写入 servo_data.csv、
指令均在 [-1,1]、且存在非平凡运动指令。

用法：python data_collection/scripts/test_open_loop_offline.py    退出码 0=通过
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from data_collection.config import Config
from data_collection.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    cfg = Config()
    # 无传感器、log-only、短时长、本地输出
    cfg.output.enable_mocap_csv = False
    cfg.output.enable_imu_csv = False
    cfg.output.enable_force_csv = False
    cfg.output.root_dir = ROOT / "data_collection" / "sessions_test"
    cfg.orchestrator.static_duration_s = 0.0
    cfg.orchestrator.calibration_duration_s = 0.0
    cfg.orchestrator.exploration_duration_s = 1.5
    cfg.servo.trajectory_type = "open_loop"
    cfg.servo.enabled = False  # log-only，不打开串口
    cfg.servo.open_loop_pretension_norm = 0.15
    cfg.servo.open_loop_fs = 100.0
    cfg.servo.open_loop_segments = [
        {"kind": "lissajous", "amp": 0.4, "f1": 0.5, "f2": 0.7, "duration_s": 1.0},
    ]

    orch = Orchestrator(cfg)
    rc = orch.run()
    if rc != 0:
        print("[FAIL] orchestrator.run() 返回非零")
        return 1

    sessions = sorted((ROOT / "data_collection" / "sessions_test").glob("session_*"))
    if not sessions:
        print("[FAIL] 未生成 session 目录")
        return 1

    csv_path = sessions[-1] / "servo_data.csv"
    df = pd.read_csv(csv_path)
    if len(df) < 10:
        print(f"[FAIL] servo_data.csv 行数过少: {len(df)}")
        return 1

    cols = ["servo_1_target_deg", "servo_2_target_deg",
            "servo_3_target_deg", "servo_4_target_deg"]
    cmd = df[cols].to_numpy(float)
    ok_range = bool(np.all((cmd >= -1.0) & (cmd <= 1.0)))
    ptp = float(np.ptp(cmd, axis=0).max())
    ok_motion = ptp > 0.05

    print(f"[{'PASS' if ok_range else 'FAIL'}] 指令均在 [-1,1]")
    print(f"[{'PASS' if ok_motion else 'FAIL'}] 存在非平凡激励 (max ptp={ptp:.3f})")
    print(f"  生成 {len(df)} 行，最新 session: {sessions[-1].name}")
    return 0 if (ok_range and ok_motion) else 1


if __name__ == "__main__":
    sys.exit(main())
