"""离线自测（无需任何硬件）：验证 PID 与舵机指令编码。

用法：python feedback_control/test_offline.py    退出码 0=通过
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feedback_control.pid import PIDController  # noqa: E402
from feedback_control.quat import quat_to_pitch_yaw  # noqa: E402
from feedback_control.servo import coupled_angles, format_command  # noqa: E402


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(("[PASS]" if cond else "[FAIL]"), name, f"({detail})" if detail else "")
    return cond


def main() -> int:
    ok = True

    # 1. 无误差 → 输出≈0
    pid = PIDController(kp=3, ki=0.3, kd=0.8, deadband=0.2, alpha=0.3)
    for _ in range(20):
        pid.calculate(0.0)
    out = pid.calculate(0.0)
    ok &= check("无误差输出≈0", abs(out) < 1e-6, f"out={out}")

    # 2. 正误差 → 正输出（反馈方向正确）
    pid2 = PIDController(kp=3, ki=0.0, kd=0.0, deadband=0.0, alpha=1.0)
    pid2.target = 10.0
    out2 = pid2.calculate(0.0)
    ok &= check("正误差→正输出", out2 > 0, f"out={out2}")

    # 3. 对侧耦合：[s1,s2,s3,s4] = [u_pitch, u_roll, -u_pitch, -u_roll]
    angles = coupled_angles(30.0, -20.0)
    ok &= check("对侧耦合正确", angles == [30, -20, -30, 20], f"{angles}")

    # 4. 协议帧格式
    cmd = format_command(angles)
    ok &= check("协议格式", cmd == b"30,-20,-30,20\n", f"{cmd!r}")

    # 5. 限幅 ±135
    angles2 = coupled_angles(999.0, 0.0)
    ok &= check("限幅 ±135", angles2 == [135, 0, -135, 0], f"{angles2}")

    # 6. 四元数 ↔ (pitch,yaw) 严格对应（回归：曾把 R[1,2] 元素写错）
    known = [
        # (qw, qx, qy, qz, 期望 pitch, 期望 yaw)
        (0.987672114351, 0.086410113286, 0.130029500652, -0.011376107231, 10.0, 15.0),
        (0.943029527380, 0.252684000300, -0.209064612935, 0.056018694202, 30.0, -25.0),
    ]
    ok6 = True
    for qw, qx, qy, qz, ep, ey in known:
        p, y = quat_to_pitch_yaw(qw, qx, qy, qz)
        err = max(abs(p - ep), abs(y - ey))
        ok6 &= err < 1e-9
        if err >= 1e-9:
            print(f"    [FAIL] quat -> (pitch={p}, yaw={y}) 期望 ({ep}, {ey})")
    ok &= check("四元数↔欧拉角往返一致", ok6)

    print("\n全部通过 ✓" if ok else "\n存在失败项 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
