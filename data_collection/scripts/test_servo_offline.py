"""舵机控制链路离线测试（无需硬件）。

IK 模式 (BasePIDController 默认):
  1. 中立位 → 四路指令全 0
  2. 无误差保持 → 输出等于同目标的开环指令 (IK 前馈)
  3. 有误差 → 反馈推高指令，且始终在 [-1, 1]
  4. 超限位 → 抛 ValueError

直接模式 (use_ik=False, 无 IK 前馈):
  5. 中立位 → 全 0 / 预紧偏置全 = bias
  6. 正向误差 → 对抗对差分 (符号与 IK 一致)
  7. 对偶对称 → 同对两舵机等量反向
  8. 大误差 → 指令限幅在 [-1, 1]，不抛错

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


def test_direct_mode() -> bool:
    """直接模式 (use_ik=False) 数值测试：无 IK 前馈，对抗对差分。"""
    ok = True

    # 1. 中立位 → 全 0
    ctrl = BasePIDController(use_ik=False)
    cmd = ctrl.update(0.0, 0.0, 0.0, 0.0, DT)
    ok &= _check("直接模式: 中立位指令全 0", np.allclose(cmd, 0.0, atol=1e-6),
                 f"{np.round(cmd, 6)}")

    # 2. 正向 pitch 误差 → servo_0 负、servo_2 正 (差分, 符号同 IK)
    ctrl.reset()
    cmd = ctrl.update(5.0, 0.0, 0.0, 0.0, DT)
    ok &= _check("直接模式: pitch 误差 → 0负2正",
                 cmd[0] < 0 and cmd[2] > 0, f"{np.round(cmd, 4)}")

    # 3. 正向 yaw 误差 → servo_1 负、servo_3 正
    ctrl.reset()
    cmd = ctrl.update(0.0, 5.0, 0.0, 0.0, DT)
    ok &= _check("直接模式: yaw 误差 → 1负3正",
                 cmd[1] < 0 and cmd[3] > 0, f"{np.round(cmd, 4)}")

    # 4. 对偶对称: 同对两舵机等量反向
    ctrl.reset()
    cmd = ctrl.update(5.0, 5.0, 0.0, 0.0, DT)
    sym = np.allclose(cmd[0], -cmd[2], atol=1e-6) and np.allclose(cmd[1], -cmd[3], atol=1e-6)
    ok &= _check("直接模式: 对偶对称", sym, f"{np.round(cmd, 4)}")

    # 5. 预紧偏置 → 中立位全 = bias
    ctrl_bias = BasePIDController(use_ik=False, direct_pretension_norm=0.2)
    cmd = ctrl_bias.update(0.0, 0.0, 0.0, 0.0, DT)
    ok &= _check("直接模式: 预紧偏置全 = bias", np.allclose(cmd, 0.2, atol=1e-6),
                 f"{np.round(cmd, 6)}")

    # 6. 大误差 → 指令限幅 [-1,1]，不抛错 (直接模式无 IK 限位)
    ctrl_clip = BasePIDController(use_ik=False, direct_gain=0.1)
    ctrl_clip.reset()
    cmd = ctrl_clip.update(100.0, 0.0, 0.0, 0.0, DT)
    ok &= _check("直接模式: 指令限幅在 [-1,1]", np.all(np.abs(cmd) <= 1.0 + 1e-9),
                 f"{np.round(cmd, 4)}")

    return ok


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
    print("----- 直接模式 (use_ik=False) -----")
    all_ok &= test_direct_mode()

    print()
    print("所有检查通过 ✓" if all_ok else "存在失败项 ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
