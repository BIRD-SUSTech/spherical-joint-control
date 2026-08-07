# 缆驱动球关节研究平台

2-DOF 缆驱动球关节（**pitch** 绕 X 轴、**yaw** 绕 Y 轴）的实验/研究平台：几何运动学与控制算法、多传感器同步采集、轨迹跟踪与误差评估，为后续数据驱动建模 / 校准 / 学习积累数据集。

## 硬件概述

物理硬件（非仿真），核心结构参数以 `control_model/geometric_ik.py` 为准：

| 参数          | 值                      |
| ----------- | ---------------------- |
| 球体半径 R      | 25 mm                  |
| 缆绳固定点（球顶圆环） | phi = 41°              |
| 球面缆绳点（球底圆环） | phi = 137°             |
| 四点方位角       | 绕 Z 轴 90° 间隔           |
| 舵盘半径        | 18 mm                  |
| 舵机总行程       | 270°（中位 135°，半行程 135°） |
| 结构限位        | ±60°                   |

传感器 / 执行器：

| 设备            | 用途            | 默认接口                       |
| ------------- | ------------- | -------------------------- |
| Nokov 光学动捕    | 球杆 6-DOF 姿态反馈 | 服务器 IP `10.1.1.198`        |
| IM948 BLE IMU | 惯性姿态 / 加速度    | BLE 地址 `A5:B2:90:FF:4A:12` |
| 六维力传感器        | 缆绳张力          | 串口 COM6，MODBUS-RTU         |
| 4× 舵机         | 缆绳驱动          | 串口 COM5，115200 baud        |

## 子系统

```
spherical_joint/
├── control_model/        # 纯算法：运动学与控制
│   ├── geometric_ik.py            # 几何逆运动学：姿态 → 4 缆长变化
│   ├── open_loop/open_loop_controller.py   # 开环控制器（纯前馈，标定用）
│   └── base_PID/pid_controller.py          # IK 前馈 + 任务空间 PID 反馈
├── data_collection/      # 硬件采集管道（详见其 README）
│   ├── main.py           # CLI 入口
│   ├── config.py         # 集中配置 (dataclass + JSON)
│   ├── orchestrator.py   # 阶段调度、多线程采集、CSV 输出
│   ├── servo_controller.py       # 舵机控制（log-only / 串口 / PID 轨迹跟踪）
│   ├── valuation.py      # 轨迹跟踪误差评估与可视化
│   ├── utils/            # 基础工具（数据类 / CSV 写入 / 会话管理 / 轨迹）
│   ├── sensor_collectors/        # mocap / IMU / force 采集器
│   └── scripts/          # 传感器独立测试脚本
└── requirements.txt      # numpy / bleak / pyserial
```

### control_model — 控制算法

- **`GeometricIK`**：给定 pitch/yaw，按球面大圆弧计算四条缆绳长度相对中立位的变化量（mm），角度超 ±60° 抛错。
- **`OpenLoopController`**：目标姿态直接经 IK 转成归一化舵机角度 [-1, 1]，纯前馈无反馈，适用于标定轨迹。
- **`BasePIDController`**：**IK 前馈 + 任务空间 PID 反馈**。PID 在 pitch/yaw 任务空间对误差闭环，输出解释为对**目标姿态的修正量**，合成后统一经 IK 转舵机量。避免求解雅可比；稳态时反馈收敛为零，不引入静态误差。

控制数据流：

```
target pitch/yaw ──→ IK 前馈 ──→ ΔL[4] ──→ 归一化舵机角度 [-1, 1]
       ▲                                 （0 = 中位）
       └── PID 修正目标姿态 ← 动捕姿态误差
```

### data_collection — 数据采集

协调 Nokov 动捕、IM948 BLE IMU、六维力传感器与舵机控制，多路数据同步落盘 CSV。

- **三阶段流程**：`STATIC`（中立位静止，IMU 自清零、捕获参考四元数）→ `CALIBRATION`（手动沿 pitch/yaw 运动 ≥ ±15°）→ `EXPLORATION`（自动执行轨迹或手动探索）。
- **多线程架构**：采集线程 → `queue.Queue` → 单一 consumer 线程 → CSV，队列满时丢弃并累计 `dropped_count`。
- **轨迹跟踪**：配置 waypoints 时自动启用 `PIDServoController`（独立线程 ~200Hz，读 mocap 姿态反馈闭环）。
- **安全默认**：`servo.enabled = false` 时 PID 指令只记录不驱动物理舵机；串口打不开自动退回 log-only。

详细安装、配置、数据格式与测试指南见 **[data_collection/README.md](data_collection/README.md)**。

### valuation.py — 轨迹跟踪评估

对齐 mocap 实测与 servo 预期数据（`merge_asof` 邻近时间戳，50ms 容差），计算 pitch/yaw 的 MAE / RMSE / 最大绝对误差，并提供：

- `plot_pitch_yaw_seperated`：pitch / yaw 跟踪曲线对比
- `plot_pitch_yaw_trajectory`：Pitch-Yaw 二维轨迹对比
- `animate_pitch_yaw_trajectory`：动态轨迹动画（可存 GIF）

## 安装

项目使用 `spherical_joint` conda 环境：

```bash
conda activate spherical_joint
pip install -r requirements.txt
```

**Nokov 动捕 SDK** 不是 pip 包，需单独安装厂商 SDK，安装后应可通过 `from nokov import nokovsdk` 导入。

## 快速开始

```bash
# 1. 离线验证控制算法（无需硬件）
python data_collection/scripts/test_servo_offline.py

# 2. 完整采集（EXPLORATION 默认无限，Ctrl+C 停止）
python -m data_collection.main run

# 3. 指定时长的自动流程
python -m data_collection.main run -c config.json --duration 30

# 4. 轨迹跟踪评估（见 valuation.py 顶部的数据路径配置）
python -m data_collection.valuation
```

## 测试

- `data_collection/scripts/test_servo_offline.py`：IK + PID + 归一化数值正确性（无需硬件）
- `scripts/test_mocap.py` / `test_imu.py` / `test_force.py`：各传感器独立连接测试
- `python -m data_collection.main run --static 5 --calib 10 --duration 10`：静止联合测试（验证采集管道与 CSV 落盘）

## 文档索引

| 文档                                                     | 内容                              |
| ------------------------------------------------------ | ------------------------------- |
| [data_collection/README.md](data_collection/README.md) | 采集系统详细文档（安装 / 配置 / 数据格式 / 测试指南） |
