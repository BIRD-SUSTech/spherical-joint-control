"""直接模式 direct_gain 标定脚本（需要真实硬件：舵机串口 + Nokov 动捕）。

原理：给对抗对施加一个已知差分 norm，用动捕测量关节实际转动角度，
     direct_gain = 施加的差分 / 实测转角 (norm/度)。

流程：
  1. 连接动捕，保持球杆中立位，捕获参考四元数
  2. 舵机回中位，读初始姿态（应接近 0,0）
  3. pitch 标定：servo_0/2 施加 ±amp，测稳态 pitch 角
  4. yaw  标定：servo_1/3 施加 ±amp，测稳态 yaw 角
  5. 输出 gain_pitch / gain_yaw 及建议的 direct_gain

用法:
    python scripts/calibrate_direct_gain.py [--port COM5] [--baudrate 115200]
        [--amp 0.05] [--bias 0.05] [--settle 2] [--sample 2]
        [--rigid-body-id 0] [--ip 10.1.1.198]

退出码: 0=成功, 1=失败.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_collection.config import Config  # noqa: E402
from data_collection.orchestrator import _quat_to_pitch_yaw  # noqa: E402
from data_collection.sensor_collectors.mocap_collector import MocapCollector  # noqa: E402
from data_collection.servo_controller import SerialServoDriver  # noqa: E402


def _latest_frame(q: queue.Queue):
    """排空队列并返回最新一帧."""
    frame = None
    while True:
        try:
            frame = q.get_nowait()
        except queue.Empty:
            break
    return frame


def _find_rb(frame, rigid_body_id):
    if frame is None:
        return None
    for rb in frame.rigid_bodies:
        if rb.id == rigid_body_id:
            return rb
    return None


def _wait_ref_quat(q, rigid_body_id, timeout_s):
    """等待目标刚体出现，返回参考四元数 [qw,qx,qy,qz] 或 None."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rb = _find_rb(_latest_frame(q), rigid_body_id)
        if rb is not None:
            return np.array([rb.qw, rb.qx, rb.qy, rb.qz])
        time.sleep(0.05)
    return None


def _measure_pose(q, ref_quat, rigid_body_id, sample_s):
    """采样 sample_s 秒，返回平均 (pitch_deg, yaw_deg) 或 None."""
    poses = []
    deadline = time.time() + sample_s
    while time.time() < deadline:
        rb = _find_rb(_latest_frame(q), rigid_body_id)
        if rb is not None:
            poses.append(_quat_to_pitch_yaw(rb.qx, rb.qy, rb.qz, rb.qw, ref_quat))
        time.sleep(0.03)
    if not poses:
        return None
    return tuple(np.asarray(poses).mean(axis=0))


def _calibrate_axis(driver, q, ref_quat, rigid_body_id, cmd, settle_s, sample_s):
    """发送指令 cmd，等待 settle 后采样，返回 (指令前姿态, 指令后姿态)."""
    driver.send(cmd)
    time.sleep(settle_s)
    return _measure_pose(q, ref_quat, rigid_body_id, sample_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="直接模式 direct_gain 标定")
    parser.add_argument("--port", type=str, default=None, help="舵机串口 (默认取配置)")
    parser.add_argument("--baudrate", type=int, default=None, help="舵机波特率")
    parser.add_argument("--ip", type=str, default=None, help="动捕服务器 IP")
    parser.add_argument("--rigid-body-id", type=int, default=0, help="动捕刚体 ID")
    parser.add_argument("--amp", type=float, default=0.05, help="差分幅度 (归一化 norm)")
    parser.add_argument("--bias", type=float, default=0.05, help="回中位时的预紧偏置 (norm)")
    parser.add_argument("--settle", type=float, default=2.0, help="指令后等待稳定 (s)")
    parser.add_argument("--sample", type=float, default=2.0, help="姿态采样窗口 (s)")
    args = parser.parse_args()

    cfg = Config()
    if args.ip:
        cfg.mocap.server_ip = args.ip
    if args.port:
        cfg.servo.port = args.port
    if args.baudrate:
        cfg.servo.baudrate = args.baudrate
    cfg.servo.rigid_body_id = args.rigid_body_id

    print("========== direct_gain 标定 ==========")
    print(f"舵机串口: {cfg.servo.port} @ {cfg.servo.baudrate}")
    print(f"动捕刚体 ID: {cfg.servo.rigid_body_id}, 差分幅度: {args.amp}, bias: {args.bias}")
    print("警告: 标定过程中舵机会带动球杆运动，请确保周围无遮挡/干涉。")
    print()

    # ---- 动捕 ----
    q: queue.Queue = queue.Queue(maxsize=2000)
    start_ev, stop_ev = threading.Event(), threading.Event()
    mocap = MocapCollector(cfg.mocap, q, start_ev, stop_ev)
    if not mocap.connect():
        print("[FAIL] 无法连接动捕服务器（请确认 Nokov 软件已启动）")
        return 1
    mocap.start()

    try:
        print("保持球杆中立位，正在捕获参考姿态...")
        ref_quat = _wait_ref_quat(q, cfg.servo.rigid_body_id, timeout_s=15)
        if ref_quat is None:
            print(f"[FAIL] 15s 内未捕获到刚体 ID {cfg.servo.rigid_body_id}"
                  "（请确认球杆 rigid body 已建好且可见）")
            return 1
        print(f"[OK] 参考四元数 = {np.round(ref_quat, 4)}")

        # ---- 舵机 ----
        driver = SerialServoDriver(port=cfg.servo.port, baudrate=cfg.servo.baudrate)
        if not driver.connect():
            print(f"[FAIL] 无法打开舵机串口 {cfg.servo.port}")
            return 1

        try:
            amp, bias = args.amp, args.bias
            neutral = np.full(4, bias)

            # 初始姿态
            driver.send(neutral)
            time.sleep(args.settle)
            p0 = _measure_pose(q, ref_quat, cfg.servo.rigid_body_id, args.sample)
            if p0 is None:
                print("[FAIL] 标定期间未收到动捕姿态")
                return 1
            print(f"[INFO] 初始姿态: pitch={p0[0]:+.2f}°, yaw={p0[1]:+.2f}°")
            print()

            # ---- pitch 标定: servo_0/2 差分 ----
            cmd_pitch = np.array([-amp + bias, bias, +amp + bias, bias])
            print(f"pitch 标定: 指令 = {np.round(cmd_pitch, 4)}")
            pp = _calibrate_axis(driver, q, ref_quat, cfg.servo.rigid_body_id,
                                 cmd_pitch, args.settle, args.sample)
            if pp is None:
                print("[FAIL] pitch 标定未收到动捕姿态")
                return 1
            dpitch = pp[0] - p0[0]
            gain_pitch = amp / dpitch if abs(dpitch) > 1e-6 else float("inf")
            print(f"  实测 pitch = {pp[0]:+.2f}° (Δ={dpitch:+.2f}°)")
            print(f"  gain_pitch = {gain_pitch:.6f}")
            if dpitch < 0:
                print("  [WARN] pitch 转动方向与预期相反！请检查舵机接线或指令符号。")

            # 回中位
            driver.send(neutral)
            time.sleep(args.settle)

            # ---- yaw 标定: servo_1/3 差分 ----
            cmd_yaw = np.array([bias, -amp + bias, bias, +amp + bias])
            print(f"\nyaw 标定: 指令 = {np.round(cmd_yaw, 4)}")
            py = _calibrate_axis(driver, q, ref_quat, cfg.servo.rigid_body_id,
                                 cmd_yaw, args.settle, args.sample)
            if py is None:
                print("[FAIL] yaw 标定未收到动捕姿态")
                return 1
            dyaw = py[1] - p0[1]
            gain_yaw = amp / dyaw if abs(dyaw) > 1e-6 else float("inf")
            print(f"  实测 yaw = {py[1]:+.2f}° (Δ={dyaw:+.2f}°)")
            print(f"  gain_yaw = {gain_yaw:.6f}")
            if dyaw < 0:
                print("  [WARN] yaw 转动方向与预期相反！请检查舵机接线或指令符号。")

            # ---- 汇总 ----
            print("\n========== 标定结果 ==========")
            print(f"gain_pitch = {gain_pitch:.6f}  (norm/度)")
            print(f"gain_yaw   = {gain_yaw:.6f}  (norm/度)")
            if np.isfinite(gain_pitch) and np.isfinite(gain_yaw):
                rec = (gain_pitch + gain_yaw) / 2
                print(f"建议 direct_gain = {rec:.6f}")
                print("配置示例: \"servo\": {\"use_ik_feedforward\": false,"
                      f" \"direct_gain\": {rec:.6f}}}")
                return 0
            print("[WARN] 某一轴转角过小，请增大 --amp 后重试。")
            return 1

        finally:
            driver.send(np.zeros(4))
            driver.disconnect()
    finally:
        mocap.stop()
        mocap.join(timeout=3.0)


if __name__ == "__main__":
    sys.exit(main())
