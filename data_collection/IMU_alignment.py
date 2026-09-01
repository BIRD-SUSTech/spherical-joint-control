"""IMU 与动捕数据对齐，可视化 IMU 相对动捕的误差。

利用 calibration 阶段数据线性拟合 IMU pitch/yaw → 动捕 pitch/yaw 的映射
（含跨轴耦合），再以动捕为真值计算 exploration 阶段的误差。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_collection.orchestrator import _quat_to_pitch_yaw, _mocap_reference_quat

TOLERANCE_NS = 50_000_000  # 50 ms


# ---------------------------------------------------------------------------
# 四元数 → pitch/yaw (各自用自己的参考四元数)
# ---------------------------------------------------------------------------


def _quat_df_to_pitch_yaw(df: pd.DataFrame, *,
                          qw_col: str, qx_col: str, qy_col: str, qz_col: str,
                          time_col: str, ref_quat: np.ndarray,
                          keep_phase: bool = False) -> pd.DataFrame:
    """将 DataFrame 中的四元数转为 pitch / yaw 角度。"""
    pitches, yaws = [], []
    for _, row in df.iterrows():
        p, y = _quat_to_pitch_yaw(
            row[qx_col], row[qy_col], row[qz_col], row[qw_col],
            ref_quat=ref_quat,
        )
        pitches.append(p)
        yaws.append(y)

    result = {"time": df[time_col], "pitch": pitches, "yaw": yaws}
    if keep_phase and "phase" in df.columns:
        result["phase"] = df["phase"]
    return pd.DataFrame(result).sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 标定：用 calibration 阶段拟合 IMU→Mocap 线性映射
# ---------------------------------------------------------------------------


def calibrate_linear(imu_path: str, mocap_path: str) -> tuple[np.ndarray, np.ndarray]:
    """利用 calibration 阶段拟合 IMU pitch/yaw → 动捕 pitch/yaw 的线性映射。

    Returns:
        (coef_p, coef_y): 各为 (a, b, c) 使得:
            mocap_pitch ≈ a*imu_p + b*imu_y + c
            mocap_yaw  ≈ a*imu_p + b*imu_y + c   (不同的 a,b,c)
    """
    imu = pd.read_csv(imu_path)
    mocap = pd.read_csv(mocap_path)

    # IMU 以自身首帧为参考（内部约定，线性拟合会吸收）；动捕用 STATIC 末帧（与实时口径一致）
    imu_ref = np.array([imu["quat_w"].iloc[0], imu["quat_x"].iloc[0],
                         imu["quat_y"].iloc[0], imu["quat_z"].iloc[0]])
    mocap_ref = _mocap_reference_quat(mocap)

    # 筛选标定阶段，按时间对齐
    mocap_cal = mocap[mocap["phase"] == "calibration"]
    if len(mocap_cal) == 0:
        raise ValueError("动捕数据中没有 calibration 阶段")

    merged = pd.merge_asof(
        imu[["pc_timestamp_ns", "quat_w", "quat_x", "quat_y", "quat_z"]],
        mocap_cal[["pc_timestamp_ns", "rigid_body_qw", "rigid_body_qx",
                    "rigid_body_qy", "rigid_body_qz"]],
        on="pc_timestamp_ns", direction="nearest", tolerance=TOLERANCE_NS,
    ).dropna()

    print(f"  Calibration 对齐: {len(merged)} 对")

    # 各自用各自参考转为 pitch/yaw
    im_p, im_y, mc_p, mc_y = [], [], [], []
    for _, row in merged.iterrows():
        mp, my = _quat_to_pitch_yaw(
            row["rigid_body_qx"], row["rigid_body_qy"],
            row["rigid_body_qz"], row["rigid_body_qw"], ref_quat=mocap_ref)
        ip, iy = _quat_to_pitch_yaw(
            row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"], ref_quat=imu_ref)
        im_p.append(ip); im_y.append(iy); mc_p.append(mp); mc_y.append(my)

    X = np.column_stack([im_p, im_y, np.ones(len(im_p))])
    coef_p, _, _, _ = np.linalg.lstsq(X, mc_p, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(X, mc_y, rcond=None)

    for name, coef, y in [("Pitch", coef_p, mc_p), ("Yaw", coef_y, mc_y)]:
        ss_res = np.sum((y - X @ coef) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"  {name}: m = {coef[0]:.4f}*imu_p + {coef[1]:.4f}*imu_y + {coef[2]:.4f}  "
              f"(R^2={r2:.4f})")

    return coef_p, coef_y


def apply_linear_calibration(imu_path: str,
                             coef_p: np.ndarray,
                             coef_y: np.ndarray) -> pd.DataFrame:
    """加载 IMU 数据，用自身参考转 pitch/yaw，再应用线性标定映射。

    Returns:
        DataFrame: time, pitch, yaw (标定后，对齐到动捕坐标系)
    """
    imu = pd.read_csv(imu_path)

    imu_ref = np.array([imu["quat_w"].iloc[0], imu["quat_x"].iloc[0],
                         imu["quat_y"].iloc[0], imu["quat_z"].iloc[0]])

    pitches, yaws = [], []
    for _, row in imu.iterrows():
        ip, iy = _quat_to_pitch_yaw(
            row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"],
            ref_quat=imu_ref)
        pitches.append(ip)
        yaws.append(iy)

    X = np.column_stack([pitches, yaws, np.ones(len(pitches))])
    return pd.DataFrame({
        "time": imu["pc_timestamp_ns"],
        "pitch": X @ coef_p,
        "yaw": X @ coef_y,
    }).sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 动捕数据加载
# ---------------------------------------------------------------------------


def load_mocap_as_pitch_yaw(path: str) -> pd.DataFrame:
    """加载动捕数据，转为 pitch / yaw（以自身首帧为参考）。"""
    df = pd.read_csv(path)
    ref_q = _mocap_reference_quat(df)
    return _quat_df_to_pitch_yaw(
        df,
        qw_col="rigid_body_qw", qx_col="rigid_body_qx",
        qy_col="rigid_body_qy", qz_col="rigid_body_qz",
        time_col="pc_timestamp_ns", ref_quat=ref_q, keep_phase=True,
    )


# ---------------------------------------------------------------------------
# 时间对齐 & 误差计算
# ---------------------------------------------------------------------------


def align(imu: pd.DataFrame, mocap: pd.DataFrame) -> pd.DataFrame:
    """按最近时间戳对齐 IMU 到动捕，仅保留 exploration 阶段。"""
    imu_r = imu.rename(columns={"pitch": "imu_pitch", "yaw": "imu_yaw"})
    mocap_r = mocap.rename(columns={"pitch": "mocap_pitch", "yaw": "mocap_yaw"})

    merged = pd.merge_asof(
        imu_r, mocap_r[["time", "mocap_pitch", "mocap_yaw", "phase"]],
        on="time", direction="nearest", tolerance=TOLERANCE_NS,
    ).dropna().reset_index(drop=True)

    merged = merged[merged["phase"] == "exploration"].reset_index(drop=True)
    merged["pitch_error"] = merged["imu_pitch"] - merged["mocap_pitch"]
    merged["yaw_error"] = merged["imu_yaw"] - merged["mocap_yaw"]
    return merged


def error_report(aligned: pd.DataFrame) -> None:
    """打印误差指标 (动捕为真值, IMU - 动捕)。"""
    for axis in ("pitch", "yaw"):
        err = aligned[f"{axis}_error"]
        abs_err = err.abs()
        print(f"========== {axis.upper()} IMU-Mocap 误差 ==========")
        print(f"  MAE:      {abs_err.mean():.4f} deg")
        print(f"  RMSE:     {np.sqrt((err ** 2).mean()):.4f} deg")
        print(f"  Max Abs:  {abs_err.max():.4f} deg")
        print(f"  Bias:     {err.mean():.4f} deg")
        print()


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def plot_comparison(aligned: pd.DataFrame) -> None:
    """IMU vs 动捕 时间序列对比 + 误差曲线。"""
    t = (aligned["time"] - aligned["time"].iloc[0]) / 1e9

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, aligned["mocap_pitch"], label="Mocap", alpha=0.85, linewidth=1.2)
    axes[0].plot(t, aligned["imu_pitch"], label="IMU (calibrated)", alpha=0.75, linewidth=1.0)
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].set_title("Pitch: IMU vs Mocap")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(t, aligned["mocap_yaw"], label="Mocap", alpha=0.85, linewidth=1.2)
    axes[1].plot(t, aligned["imu_yaw"], label="IMU (calibrated)", alpha=0.75, linewidth=1.0)
    axes[1].set_ylabel("Yaw (deg)")
    axes[1].set_title("Yaw: IMU vs Mocap")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[2].plot(t, aligned["pitch_error"], label="Pitch Error", alpha=0.8, linewidth=0.8)
    axes[2].plot(t, aligned["yaw_error"], label="Yaw Error", alpha=0.8, linewidth=0.8)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.6)
    axes[2].set_xlabel("Time (s, relative)")
    axes[2].set_ylabel("Error (deg)")
    axes[2].set_title("IMU - Mocap Error")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_time_series(aligned: pd.DataFrame) -> None:
    """Pitch / Yaw 方向动捕和 IMU 测量值随时间变化。"""
    t = (aligned["time"] - aligned["time"].iloc[0]) / 1e9

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for ax, axis in zip(axes, ("pitch", "yaw")):
        ax.plot(t, aligned[f"mocap_{axis}"], label="Mocap",
                linewidth=1.4, alpha=0.9, color="#2c7bb6")
        ax.plot(t, aligned[f"imu_{axis}"], label="IMU (calibrated)",
                linewidth=1.0, alpha=0.8, color="#d7191c")
        ax.set_ylabel(f"{axis.title()} (deg)")
        ax.set_title(f"{axis.title()}: Mocap vs IMU")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_error_distribution(aligned: pd.DataFrame) -> None:
    """误差分布直方图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, axis in zip(axes, ("pitch", "yaw")):
        err = aligned[f"{axis}_error"]
        ax.hist(err, bins=60, alpha=0.7, edgecolor="white")
        ax.axvline(err.mean(), color="red", linestyle="--",
                   label=f"Mean = {err.mean():.3f} deg")
        ax.set_xlabel("Error (deg)")
        ax.set_ylabel("Count")
        ax.set_title(f"{axis.upper()} Error Distribution")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_trajectory_2d(aligned: pd.DataFrame) -> None:
    """Pitch-Yaw 2D 轨迹对比。"""
    plt.figure(figsize=(10, 8))

    plt.plot(aligned["mocap_pitch"], aligned["mocap_yaw"],
             label="Mocap", linewidth=1.8, alpha=0.85)
    plt.plot(aligned["imu_pitch"], aligned["imu_yaw"],
             label="IMU (calibrated)", linewidth=1.5, alpha=0.75)

    plt.scatter(aligned["mocap_pitch"].iloc[0], aligned["mocap_yaw"].iloc[0],
                marker="o", s=80, label="Start (Mocap)")
    plt.scatter(aligned["imu_pitch"].iloc[0], aligned["imu_yaw"].iloc[0],
                marker="o", s=80, label="Start (IMU)")

    plt.xlabel("Pitch (deg)")
    plt.ylabel("Yaw (deg)")
    plt.title("Pitch-Yaw Trajectory: IMU vs Mocap")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SESSION_DIR = r"\\BIRD-NAS\shared_data\spherical_joint\data_collection\session_20260807_172652"
    imu_csv = f"{SESSION_DIR}/imu_data.csv"
    mocap_csv = f"{SESSION_DIR}/mocap_data.csv"

    print("=" * 60)
    print("Step 1: 线性标定 (calibration 阶段)")
    coef_p, coef_y = calibrate_linear(imu_csv, mocap_csv)

    print()
    print("Step 2: 应用标定到 IMU ...")
    imu = apply_linear_calibration(imu_csv, coef_p, coef_y)
    print(f"  IMU: {len(imu)} 行")

    print("  加载动捕 ...")
    mocap = load_mocap_as_pitch_yaw(mocap_csv)
    print(f"  Mocap: {len(mocap)} 行")

    print()
    print("Step 3: 时间对齐 & 误差计算 (exploration)")
    aligned = align(imu, mocap)
    print(f"  Exploration 对齐: {len(aligned)} 行\n")
    error_report(aligned)

    print("Step 4: 可视化 ...")
    plot_comparison(aligned)
    plot_error_distribution(aligned)
    plot_trajectory_2d(aligned)
