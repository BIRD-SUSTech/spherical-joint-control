"""离线自测：加载 g，验证前馈控制器与 PID 逻辑（无需硬件）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feedback_control.feedforward import DiffPID, FeedforwardController, decode


def main() -> int:
    ckpt = Path("feedforward/outputs/g_direct_v1/g_checkpoint.pt")
    if not ckpt.exists():
        print(f"[SKIP] 未找到 {ckpt}")
        return 0

    ffc = FeedforwardController(str(ckpt), device="cpu")
    dq_d = np.array([0.1, -0.05])
    q_meas = np.array([5.0, -3.0])
    gyro = np.array([1.0, -2.0, 0.5])
    force = np.array([8000.0, 9000.0, 7000.0, 8500.0])

    d = ffc.compute(dq_d, q_meas, gyro, force)
    u = ffc.u_ff(dq_d, q_meas, gyro, force)
    print("差分 d =", np.round(d, 4))
    print("u_ff   =", np.round(u, 4))
    ok = d.shape == (2,) and np.isfinite(d).all() and np.all(np.abs(u) <= 1.0 + 1e-6)
    print(f"[{'PASS' if ok else 'FAIL'}] 前馈输出 shape/范围")

    # 解码正确性：d=0 -> 全预紧
    u0 = decode(np.zeros(2), 0.15)
    ok0 = np.allclose(u0, [0.15, 0.15, 0.15, 0.15])
    print(f"[{'PASS' if ok0 else 'FAIL'}] decode(d=0) = 预紧")

    # PID 方向：正 pitch 误差 -> d1<0；负 yaw 误差 -> d2>0
    pid = DiffPID(kp_pitch=0.04, kp_yaw=0.04)
    d_fb = pid.update(5.0, -3.0, 0.01)
    ok1 = d_fb[0] < 0 and d_fb[1] > 0
    print(f"[{'PASS' if ok1 else 'FAIL'}] PID 差分方向 (d_fb={np.round(d_fb,4)})")

    return 0 if (ok and ok0 and ok1) else 1


if __name__ == "__main__":
    sys.exit(main())
