"""轨迹跟踪误差评估与可视化。

两种数据来源（自动识别）：
- 双文件模式：servo_data.csv（目标 target_pitch/target_yaw）+ mocap_data.csv（实测，四元数转姿态）
- 单文件模式：feedback_control 日志（target_* 为目标、current_* 为动捕实测）

用法：
    # 单文件：feedback_control 日志
    python -m data_collection.valuation --servo feedback_control/logs/closed_loop_xxx.csv

    # 双文件：data_collection 会话
    python -m data_collection.valuation --servo <session>/servo_data.csv --mocap <session>/mocap_data.csv
"""

from data_collection.orchestrator import _quat_to_pitch_yaw, _mocap_reference_quat
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

MOTION_CAP_PATH = r"\\BIRD-NAS\shared_data\spherical_joint\data_collection\session_20260806_171033\mocap_data.csv"
SERVO_DATA_PATH = r"\\BIRD-NAS\shared_data\spherical_joint\data_collection\session_20260806_171033\servo_data.csv"
TOLERANCE = 50_000_000  # 单位：ns 这里是50毫秒


def _pick_col(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    """按候选名依次取列，找不到返回 None。"""
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _pick_time_col(df: pd.DataFrame) -> str:
    """时间列：优先 pc_timestamp_ns（data_collection），其次 t_s（feedback_control）。"""
    for n in ("pc_timestamp_ns", "t_s"):
        if n in df.columns:
            return n
    return df.columns[0]


def evaluation(mocap_path: str = None, servo_path: str = None) -> pd.DataFrame | None:
    """轨迹跟踪误差评估。

    Args:
        mocap_path: mocap CSV 路径（双文件模式）；None = 单文件模式
            （servo_path 为 feedback_control 日志，current_* 即动捕实测）。
        servo_path: servo CSV 路径（servo_data.csv 或 feedback_control 日志）。

    Returns:
        DataFrame: time, servo_pitch, servo_yaw, mocap_pitch, mocap_yaw,
                   pitch_error, yaw_error (servo_* = 目标, mocap_* = 实测).
    """
    if servo_path is None:
        print("请提供 servo csv 路径 (servo_data.csv 或 feedback_control 日志)")
        return None

    servo_df = pd.read_csv(servo_path)
    time_col = _pick_time_col(servo_df)

    # 目标姿态（新列名，兼容旧会话的 joint_angle_*_deg）
    target_pitch = _pick_col(servo_df, ["target_pitch", "joint_angle_1_deg"])
    target_yaw = _pick_col(servo_df, ["target_yaw", "joint_angle_2_deg"])
    if target_pitch is None or target_yaw is None:
        print("servo csv 缺少目标姿态列 (target_pitch/target_yaw 或旧名 joint_angle_*_deg)")
        return None

    if mocap_path is None:
        # ---- 单文件模式：feedback_control 日志 ----
        current_pitch = _pick_col(servo_df, ["current_pitch", "pitch"])
        current_yaw = _pick_col(servo_df, ["current_yaw", "yaw"])
        if current_pitch is None or current_yaw is None:
            print("单文件模式需要 current_pitch/current_yaw 列")
            return None
        print(f"[单文件模式] {servo_path}  (current_* 视为动捕实测)")
        aligned = pd.DataFrame({
            "time": servo_df[time_col],
            "servo_pitch": target_pitch,
            "servo_yaw": target_yaw,
            "mocap_pitch": current_pitch,
            "mocap_yaw": current_yaw,
        }).sort_values("time").reset_index(drop=True)
    else:
        # ---- 双文件模式：mocap + servo ----
        print(f"[双文件模式] servo={servo_path}, mocap={mocap_path}")
        mocap_df = pd.read_csv(mocap_path)
        if time_col not in mocap_df.columns:
            print(f"mocap csv 缺少时间列 {time_col}")
            return None

        # 参考坐标系：STATIC 段末帧四元数（与实时采集口径一致）
        ref_q = _mocap_reference_quat(mocap_df)

        mocap_pitch_list = []
        mocap_yaw_list = []
        for _, row in mocap_df.iterrows():
            pitch, yaw = _quat_to_pitch_yaw(
                row["rigid_body_qx"], row["rigid_body_qy"],
                row["rigid_body_qz"], row["rigid_body_qw"],
                ref_quat=ref_q,
            )
            mocap_pitch_list.append(pitch)
            mocap_yaw_list.append(yaw)

        mocap_clean = pd.DataFrame({
            "time": mocap_df[time_col],
            "mocap_pitch": mocap_pitch_list,
            "mocap_yaw": mocap_yaw_list,
            "mocap_state": mocap_df["phase"] if "phase" in mocap_df.columns else "exploration",
        }).sort_values("time")

        servo_clean = pd.DataFrame({
            "time": servo_df[time_col],
            "servo_pitch": target_pitch,
            "servo_yaw": target_yaw,
        }).sort_values("time")

        # 数据戳对齐（基于邻近策略）
        aligned = pd.merge_asof(
            left=servo_clean,
            right=mocap_clean,
            on="time",
            direction="nearest",
            tolerance=TOLERANCE,
        )
        aligned = aligned.dropna().reset_index(drop=True)
        if "mocap_state" in aligned.columns:
            aligned = aligned[aligned["mocap_state"] == "exploration"].reset_index(drop=True)

    # 误差计算（实测值 - 目标值）
    aligned["pitch_error"] = aligned["mocap_pitch"] - aligned["servo_pitch"]
    aligned["yaw_error"] = aligned["mocap_yaw"] - aligned["servo_yaw"]

    return aligned


def error_calculate(aligned_data: pd.DataFrame):

    pitch_abs_err = aligned_data["pitch_error"].abs()
    yaw_abs_err = aligned_data["yaw_error"].abs()

    print("========== 误差评估分析报告 ==========")
    print(f"Pitch 平均绝对误差 (MAE): {pitch_abs_err.mean():.4f}")
    print(f"Pitch 均方根误差 (RMSE): {np.sqrt((pitch_abs_err ** 2).mean()):.4f}")
    print(f"Pitch 最大绝对误差:      {pitch_abs_err.max():.4f}")
    print("-" * 38)
    print(f"Yaw 平均绝对误差 (MAE):   {yaw_abs_err.mean():.4f}")
    print(f"Yaw 均方根误差 (RMSE):   {np.sqrt((yaw_abs_err ** 2).mean()):.4f}")
    print(f"Yaw 最大绝对误差:        {yaw_abs_err.max():.4f}")
    print("===================================")


def save_data(aligned_data: pd.DataFrame, path: str = None):
    if path is None:
        path = f"./data_collection/outputs/output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    aligned_data.to_csv(path, index=False)
    print(f"对齐数据已保存: {path}")


def plot_pitch_yaw_seperated(aligned_data: pd.DataFrame):

    # 结果可视化
    plt.figure(figsize=(12, 8))

    # Pitch 跟踪曲线
    plt.subplot(2, 1, 1)
    plt.plot(
        aligned_data["time"],
        aligned_data["servo_pitch"],
        label="Expected Pitch (Servo)",
        linestyle="--",
    )
    plt.plot(
        aligned_data["time"],
        aligned_data["mocap_pitch"],
        label="Actual Pitch (Mocap)",
        alpha=0.7,
    )
    plt.title("Pitch Tracking: Expected vs Actual")
    plt.ylabel("Angle")
    plt.legend()

    # Yaw 跟踪曲线
    plt.subplot(2, 1, 2)
    plt.plot(
        aligned_data["time"],
        aligned_data["servo_yaw"],
        label="Expected Yaw (Servo)",
        linestyle="--",
    )
    plt.plot(
        aligned_data["time"],
        aligned_data["mocap_yaw"],
        label="Actual Yaw (Mocap)",
        alpha=0.7,
    )
    plt.title("Yaw Tracking: Expected vs Actual")
    plt.xlabel("Timestamp")
    plt.ylabel("Angle")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_pitch_yaw_trajectory(aligned_data: pd.DataFrame):
    """
    绘制以 Pitch 为横坐标，Yaw 为纵坐标的二维运动轨迹对比图
    """
    if aligned_data.empty:
        print("警告: 传入的数据为空，无法绘制轨迹图。")
        return

    plt.figure(figsize=(10, 8))

    # 绘制舵机预期轨迹 (Expected)
    # 以 servo_pitch 为 X 轴，servo_yaw 为 Y 轴
    plt.plot(
        aligned_data["servo_pitch"],
        aligned_data["servo_yaw"],
        label="Expected Trajectory (Servo)",
        linestyle="-",  # 预期轨迹使用虚线
        color="blue",
        alpha=0.8,  # 稍微增加透明度，防止完全遮挡实际轨迹
        linewidth=2,  # 增加线宽，让轨迹更清晰
    )

    # 绘制动捕实际观测轨迹 (Actual)
    # 以 mocap_pitch 为 X 轴，mocap_yaw 为 Y 轴
    plt.plot(
        aligned_data["mocap_pitch"],
        aligned_data["mocap_yaw"],
        label="Actual Trajectory (Mocap)",
        linestyle="-",  # 实际轨迹使用实线
        color="orange",
        alpha=0.8,
        linewidth=2,
    )

    # 标注起始点 (可选：有助于观察运动是从哪里开始的)
    plt.scatter(
        aligned_data["servo_pitch"].iloc[0],
        aligned_data["servo_yaw"].iloc[0],
        color="blue",
        marker="o",
        s=100,
        label="Start (Expected)",
    )
    plt.scatter(
        aligned_data["mocap_pitch"].iloc[0],
        aligned_data["mocap_yaw"].iloc[0],
        color="orange",
        marker="o",
        s=100,
        label="Start (Actual)",
    )

    # 图表装饰
    plt.title("Pitch-Yaw Tracking Trajectory", fontsize=14)
    plt.xlabel("Pitch Angle (deg)", fontsize=12)
    plt.ylabel("Yaw Angle (deg)", fontsize=12)

    # 增加网格线，方便在二维空间里估读误差距离
    plt.grid(True, linestyle=":", alpha=0.7)

    # 确保 X 轴和 Y 轴的比例是 1:1 (可选：如果您希望图形不被拉伸，体现真实的物理空间比例)
    # plt.axis('equal')

    plt.legend()
    plt.tight_layout()
    plt.show()


def animate_pitch_yaw_trajectory(aligned_data: pd.DataFrame, save_gif_path: str = None):
    """
    绘制以 Pitch 为横坐标，Yaw 为纵坐标的动态轨迹动画
    """
    if aligned_data.empty:
        print("警告: 传入的数据为空，无法绘制轨迹图。")
        return

    # 提取所需数据转换为 Numpy 数组，加快动画渲染速度
    x_exp = aligned_data["servo_pitch"].values
    y_exp = aligned_data["servo_yaw"].values
    x_act = aligned_data["mocap_pitch"].values
    y_act = aligned_data["mocap_yaw"].values

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))

    # ---------------- 关键步骤 1：固定坐标轴范围 ----------------
    # 找到所有数据的边界并留出 10% 的余量，防止动画播放时画布疯狂缩放
    x_min, x_max = min(x_exp.min(), x_act.min()), max(x_exp.max(), x_act.max())
    y_min, y_max = min(y_exp.min(), y_act.min()), max(y_exp.max(), y_act.max())

    pad_x = (x_max - x_min) * 0.1 if (x_max - x_min) != 0 else 1
    pad_y = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # ---------------- 关键步骤 2：初始化图元对象 ----------------
    # 轨迹线 (开始是空的)
    (line_exp,) = ax.plot(
        [],
        [],
        label="Expected (Servo)",
        linestyle="--",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )
    (line_act,) = ax.plot(
        [],
        [],
        label="Actual (Mocap)",
        linestyle="-",
        color="orange",
        linewidth=2,
        alpha=0.8,
    )

    # 当前点的“车头”标记 (引导点)
    (head_exp,) = ax.plot([], [], "bo", markersize=8)
    (head_act,) = ax.plot([], [], "o", color="orange", markersize=8)

    # 装饰图表
    ax.set_title("Pitch-Yaw Tracking Trajectory Animation", fontsize=14)
    ax.set_xlabel("Pitch Angle (deg)", fontsize=12)
    ax.set_ylabel("Yaw Angle (deg)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right")

    # ---------------- 关键步骤 3：定义更新函数 ----------------
    # 决定动画播放速度，如果点太多（如700多个点），每次步进可以设为 2 或 3
    step = 3

    def update(frame):
        # 截取从头到当前帧的数据
        idx = min(frame, len(x_exp) - 1)

        # 更新轨迹线
        line_exp.set_data(x_exp[: idx + 1], y_exp[: idx + 1])
        line_act.set_data(x_act[: idx + 1], y_act[: idx + 1])

        # 更新“车头”引导点
        head_exp.set_data([x_exp[idx]], [y_exp[idx]])
        head_act.set_data([x_act[idx]], [y_act[idx]])

        return line_exp, line_act, head_exp, head_act

    # ---------------- 关键步骤 4：生成并运行动画 ----------------
    frames = range(0, len(x_exp), step)

    # interval 是每帧刷新间隔的毫秒数。interval=20 约等于 50fps
    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=20, blit=True, repeat=False
    )

    # ---------------- 附加功能：保存为 GIF ----------------
    if save_gif_path:
        print(f"正在保存动画到 {save_gif_path}，请稍候...")
        # 需要确保 Python 环境里安装了 pillow (pip install pillow)
        ani.save(save_gif_path, writer="pillow", fps=30)
        print("动画保存完成！")

    plt.tight_layout()
    plt.show()  # 弹出动态窗口展示


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="轨迹跟踪误差评估与可视化")
    parser.add_argument("--servo", default=SERVO_DATA_PATH,
                        help="servo_data.csv 或 feedback_control 日志 csv 路径")
    parser.add_argument("--mocap", default=None,
                        help="mocap_data.csv 路径；省略则为单文件模式 (feedback_control 日志)")
    parser.add_argument("--plot", choices=("separated", "trajectory", "animate", "none"),
                        default="separated", help="可视化方式")
    parser.add_argument("--save", default=None, help="对齐结果另存 csv 路径")
    parser.add_argument("--gif", default=None, help="--plot animate 时 GIF 保存路径")
    args = parser.parse_args()

    # 双文件模式仅当显式给出 --mocap，或默认 NAS 路径真实存在时启用
    mocap_path = args.mocap
    if mocap_path is None:
        mocap_path = MOTION_CAP_PATH if os.path.exists(MOTION_CAP_PATH) else None

    aligned = evaluation(mocap_path, args.servo)
    if aligned is None or aligned.empty:
        print("无有效对齐数据，退出")
        raise SystemExit(1)

    error_calculate(aligned)
    if args.save:
        save_data(aligned, args.save)

    if args.plot == "none":
        pass
    elif args.plot == "trajectory":
        plot_pitch_yaw_trajectory(aligned)
    elif args.plot == "animate":
        animate_pitch_yaw_trajectory(aligned, save_gif_path=args.gif)
    else:
        plot_pitch_yaw_seperated(aligned)
