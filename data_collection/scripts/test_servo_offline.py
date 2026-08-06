"""舵机控制链路离线测试（无需硬件）。

验证 IK 前馈 + PID 反馈 + 归一化输出的数值正确性:
  1. 中立位 → 四路指令全 0
  2. 无误差保持 → 输出等于同目标的开环指令 (IK 前馈)
  3. 有误差 → 反馈推高指令，且始终在 [-1, 1]
  4. 超限位 → 抛 ValueError

用法:
    python scripts/test_servo_offline.py

退出码: 0=全部通过, 1=有失败.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from control_model.base_PID.pid_controller import BasePIDController  # noqa: E402
from control_model.open_loop.open_loop_controller import OpenLoopController  # noqa: E402

DT = 0.01


def _check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def main() -> int:
    ctrl = BasePIDController()
    open_loop = OpenLoopController()
    all_ok = True

    # 1. 中立位 → 全 0
    cmd = ctrl.update(0.0, 0.0, 0.0, 0.0, DT)
    all_ok &= _check("中立位指令全 0", np.allclose(cmd, 0.0, atol=1e-6), f"{cmd}")

    # 预热滤波器：多次调用让滤波收敛到目标姿态
    for _ in range(30):
        ctrl.update(10.0, 0.0, 10.0, 0.0, DT)

    # 2. 无误差保持 → 等于开环指令 (滤波器已收敛)
    cmd_hold = ctrl.update(10.0, 0.0, 10.0, 0.0, DT)
    ol = open_loop.command(10.0, 0.0)
    all_ok &= _check("无误差保持 ≈ 开环前馈", np.allclose(cmd_hold, ol, atol=5e-3),
                     f"hold={np.round(cmd_hold, 4)}, open_loop={np.round(ol, 4)}")

    # 3. 有误差 → 反馈推大指令，且限幅内
    # 先复位滤波器到新姿态
    ctrl.reset()
    for _ in range(30):
        ctrl.update(10.0, 0.0, 10.0, 0.0, DT)
    cmd_hold = ctrl.update(10.0, 0.0, 10.0, 0.0, DT)
    cmd_err = ctrl.update(10.0, 0.0, 9.0, 0.0, DT)
    err_norm = float(np.abs(cmd_err).max())
    fb_direction = float(np.abs(cmd_err).max()) > float(np.abs(cmd_hold).max())
    all_ok &= _check("误差反馈增大指令", fb_direction,
                     f"hold={np.abs(cmd_hold).max():.4f}, err={err_norm:.4f}")
    all_ok &= _check("指令在 [-1,1] 内", err_norm <= 1.0, f"max|cmd|={err_norm:.4f}")

    # 4. 超限位抛错
    try:
        ctrl.ik.solve(70.0, 0.0)
        all_ok &= _check("超 ±60° 抛 ValueError", False)
    except ValueError:
        all_ok &= _check("超 ±60° 抛 ValueError", True)

    print()
    print("所有检查通过 ✓" if all_ok else "存在失败项 ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
