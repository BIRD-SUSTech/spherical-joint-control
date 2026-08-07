# data_collection — 球关节数据采集系统

线驱动球关节（2-DOF：pitch 绕 X、yaw 绕 Y）的数据采集与轨迹跟踪执行模块。
协调 Nokov 光学动捕、IM948 BLE IMU、绳张力传感器、舵机控制，多路数据同步落盘为 CSV。

## 目录结构

```
data_collection/
├── main.py                 # CLI 入口
├── config.py               # 集中配置 (dataclass + JSON 加载/导出)
├── orchestrator.py         # 总协调器：阶段管理、线程调度、CSV 输出
├── servo_controller.py     # 舵机控制模块（四类控制器，见下）
├── scripts/                # 传感器独立测试脚本
├── utils/                  # 基础工具（无跨模块依赖）
│   ├── data_types.py       # 统一数据类 + CSV 列定义
│   ├── csv_writer.py       # 线程安全 CSV 写入
│   ├── session.py          # 会话目录管理
│   └── trajectory.py       # 轨迹定义与生成器
└── sensor_collectors/      # 硬件采集器（依赖 utils）
    ├── mocap_collector.py  # Nokov 动捕
    ├── imu_collector.py    # IM948 BLE IMU
    ├── force_collector.py  # 六维力传感器
    └── modbus_client.py    # MODBUS-RTU 协议客户端
```

依赖方向：`utils` ← `sensor_collectors` ← 顶层业务层（config / orchestrator / servo_controller），无反向依赖。

## 安装

项目使用 `spherical_joint` conda 环境：

```bash
conda activate spherical_joint
pip install -r requirements.txt   # numpy / bleak / pyserial
```

**Nokov 动捕 SDK** 不是 pip 包，需单独安装厂商 SDK，安装后应可通过 `from nokov import nokovsdk` 导入。

## 配置

所有参数集中在 `config.py` 的 dataclass 中，默认值即当前硬件配置：

| 配置块            | 关键字段                                                                  | 默认                         |
| -------------- | --------------------------------------------------------------------- | -------------------------- |
| `mocap`        | `server_ip`                                                           | `10.1.1.198`               |
| `imu`          | `device_address`                                                      | `A5:B2:90:FF:4A:12`        |
| `force`        | `serial_port`, `slave_address`                                        | `COM6`, `0x01`             |
| `servo`        | `trajectory_type`, `use_ik_feedforward`, `direct_gain`, `enabled`    | `idle`, `true`, `0.01`, `False` |
| `output`       | `root_dir`, 各传感器 CSV 开关                                               | `data_collection/sessions` |
| `orchestrator` | 三阶段时长、超时                                                              | —                          |

### 三种配置方式（优先级从低到高）

1. **代码默认值**：直接用 `Config()`

2. **JSON 文件**：导出模板后修改，运行时用 `-c` 指定
   
   ```python
   from data_collection.config import Config
   Config().to_json('config.json')
   ```
   
   ```bash
   python -m data_collection.main run -c config.json
   ```

3. **CLI 覆盖**：`--static/--calib/--duration/--mocap-ip/--force-port/--no-mocap/--no-imu/--no-force`

### 启用轨迹跟踪

`servo.trajectory_type = "waypoints"` 且 `trajectory_waypoints` 非空时，自动使用 `PIDServoController`（闭环）替代 `LogServoController`（仅记录）：

```json
{
  "servo": {
    "trajectory_type": "waypoints",
    "trajectory_waypoints": [[0, 0, 5], [10, 0, 5], [0, 0, 5]],
    "rigid_body_id": 0
  }
}
```

每个 waypoint 为 `[pitch_deg, yaw_deg, duration_s]`。轨迹仅在 **EXPLORATION 阶段**执行，故
`orchestrator.exploration_duration_s` 需 ≥ 轨迹总时长（`sum(duration_s)`）。

## 用法

```bash
# 完整采集（EXPLORATION 默认无限，Ctrl+C 停止）
python -m data_collection.main run

# 指定时长的自动流程
python -m data_collection.main run -c config.json --duration 30
```

### 采集阶段

| 阶段              | 默认时长   | 操作要求                         |
| --------------- | ------ | ---------------------------- |
| **STATIC**      | 3s     | 保持球杆中立位静止；IMU 自清零、动捕捕获参考四元数  |
| **CALIBRATION** | 15s    | 手动沿 pitch 和 roll 方向运动 ≥ ±15° |
| **EXPLORATION** | ∞ / 指定 | 自动执行轨迹或手动探索                  |

### 输出

每次会话创建 `data_collection/sessions/session_YYYYMMDD_HHMMSS/`，包含：

- `mocap_data.csv` / `imu_data.csv` / `force_data.csv` / `servo_data.csv`
- `session_metadata.json`：会话时间、各流 `row_counts`、`dropped_frames`、配置快照

## 数据格式

所有 CSV 行带 `phase` 列（`static` / `calibration` / `exploration`），`servo_data.csv` 除外。

- **mocap**：`pc_timestamp_ns`、`pc_receive_unix_time_ms`、`frame_index`、`hardware_timestamp`、`rigid_body_id`、位置 `x/y/z`、四元数 `qx/qy/qz/qw`
- **imu**：时间戳、`subscribe_tag`、三轴加速度（含/不含重力）、陀螺仪、磁力计、四元数、欧拉角 `angle_x/y/z`、温度气压等
- **force**：时间戳 + `ch1`–`ch6` 六通道力
- **servo**：时间戳、`servo_1..4_target_deg`（归一化 [-1,1]）、`cable_1..4_length_mm`、`joint_angle_1/2_deg`（目标/实际姿态）

## 架构

```
采集线程 (Mocap/IMU/Force) ──▶ 队列 ──▶ Consumer 线程 ──▶ CSV
PID 控制线程 (轨迹跟踪时)  ─┘                        │
                                                    ▼
                                        session_metadata.json
```

- 每个采集器在独立 daemon 线程运行，数据经 `queue.Queue` 传递给单一 consumer 线程，避免并发写文件。
- 队列满时丢弃并累计 `dropped_count`（见 metadata）。
- PID 控制线程 ~200Hz：读 mocap 姿态反馈 → `BasePIDController`（IK 前馈 + 任务空间 PID，或 `use_ik_feedforward=false` 直接差分）→ 舵机指令。

## 舵机控制模块

`servo_controller.py` 提供四个类：

| 类                       | 说明                                                     |
| ----------------------- | ------------------------------------------------------ |
| `ServoController` (ABC) | 多舵机抽象接口：`connect/disconnect/set_angles/emergency_stop` |
| `LogServoController`    | 无硬件日志桩：仅将目标角度写入输出队列                                    |
| `SerialServoDriver`     | 串口舵机驱动：向串口发送四路归一化角度，实现 `send`（兼容 `command_sink`）       |
| `PIDServoController`    | PID 闭环轨迹跟踪：独立线程执行 waypoints，读 mocap 姿态反馈               |

### 串口输出协议

`SerialServoDriver` 接收归一化舵机指令 [-1, 1]，映射为舵机角度（度）后发送：
**四个逗号分隔的整数，范围 [-135, 135]，以 `\n` 结尾**：

```
68,-68,122,0\n
```

映射关系：`deg = round(norm × 135)`（四舍五入为整数）。归一化值先裁剪到 [-1, 1] 再映射；半行程 135°（`half_range_deg` 可调）。

### 启用舵机驱动

PID 轨迹跟踪时，设置 `servo.enabled = true` 且配置 `servo.port`，编排器会自动创建
`SerialServoDriver` 并作为 PID 的 `command_sink`，物理驱动舵机：

```json
{
  "servo": {
    "enabled": true,
    "port": "COM3",
    "trajectory_type": "waypoints",
    "trajectory_waypoints": [[0, 0, 5], [10, 0, 5], [0, 0, 5]]
  }
}
```

> 默认 `enabled = false`，此时 PID 计算的舵机指令**只记录到 servo CSV，不驱动物理舵机**（安全默认）。
> 舵机串口无法打开时自动退回 log-only 模式。

### PID 工作模式（IK 前馈 vs 直接差分）

`BasePIDController` 支持两种模式，由 `servo.use_ik_feedforward` 切换：

| 模式 | 控制方式 | 适用场景 |
| --- | --- | --- |
| `use_ik_feedforward = true`（默认） | PID 输出修正目标姿态，经 `GeometricIK` 转缆长再转舵机 | 几何 IK 模型与实际运动学一致时 |
| `use_ik_feedforward = false` | PID 输出**直接**映射为对抗对差分（pitch→servo_0/2, yaw→servo_1/3），不依赖 IK 模型 | IK 模型与实际运动学偏差较大时 |

直接模式示例：

```json
{
  "servo": {
    "use_ik_feedforward": false,
    "direct_gain": 0.01,
    "direct_pretension_norm": 0.047
  }
}
```

- **`direct_gain`**：关节 1° 误差需要多少归一化舵机量。默认 `0.01` ≈ 中立位 IK 增益（1°≈0.42mm 缆长）。理论公式 `r_eff(mm) × 0.000412`，推荐用标定脚本实测（见测试指南 §4）。
- **`direct_pretension_norm`**：中立位预紧偏置，保证对偶缆绳绷紧。`0.047` ≈ 2mm 预紧的归一化等效值。
- 直接模式的**对偶约束天然满足**：同一对两舵机始终等量反向，不会出现一侧拉满另一侧完全松脱。
- 纯反馈控制、无模型前馈 → 匀速轨迹会有固有跟踪滞后，靠 PID 积分消除稳态误差。

## 测试指南

独立测试脚本位于 `data_collection/scripts/`，均无需额外参数，退出码 0=通过。

### 1. 离线测试（无需硬件，先行验证）

```bash
python data_collection/scripts/test_servo_offline.py
```

验证 IK + PID + 归一化的数值正确性（中立位全 0、无误差保持等于开环前馈、指令限幅、超 ±60° 抛错）。

### 2. 传感器独立连接测试

```bash
python data_collection/scripts/test_mocap.py [--ip 10.1.1.198] [--duration 5]
python data_collection/scripts/test_imu.py   [--timeout 20]
python data_collection/scripts/test_force.py [--port COM6] [--debug]
```

- **mocap**：确认能连服务器、有帧、可见目标刚体（球杆需预先建好 rigid body，ID 与 `servo.rigid_body_id` 一致）
- **imu**：BLE 扫描+连接+自清零约需 15s，确认收到 ~50Hz 数据包
- **force**：连续读 6 通道；超时可用 `--debug` 查看原始帧定位（串口/RS485/从机地址）

### 3. 联合测试

**静止连接测试**（验证采集管道 + CSV 落盘）：

```bash
python -m data_collection.main run --static 5 --calib 10 --duration 10
```

检查会话目录下 4 个 CSV 与 `session_metadata.json`（各流 `row_counts > 0`）。
无轨迹时 `servo_data.csv` 只有表头属正常现象。

**轨迹跟踪测试**（启用 PID 闭环）：

```bash
python -m data_collection.main run -c data_collection/scripts/test_config.json
```

验证 `servo_data.csv` 按 ~200Hz 写入、`joint_angle_*` 中目标与实际姿态逼近。
**必须开启 mocap**——PID 依赖 mocap 姿态反馈；`--no-mocap` 时反馈恒为 0，指令会推满。

### 4. direct_gain 标定（需要舵机串口 + 动捕）

直接模式（`use_ik_feedforward = false`）下，`direct_gain` 表示"关节转 1° 需要多少归一化舵机量"。
用标定脚本实测真实增益：

```bash
python data_collection/scripts/calibrate_direct_gain.py --port COM5 --amp 0.05 --bias 0.05
```

流程：捕获中立参考四元数 → 给 pitch 对抗对施加 `±amp` 差分 → 动捕实测稳态转角 →
计算 `gain = amp / 转角`；yaw 轴同理。脚本输出 `gain_pitch` / `gain_yaw` /
建议 `direct_gain`（两轴均值）及可直接粘贴的 JSON 配置示例。

- `--amp` 默认 0.05（约 5° 关节运动，安全）；`--bias` 默认 0.05 保持缆绳预紧
- 标定期间球杆会运动，请确认无干涉；结束后舵机自动回中位
- 若某轴转动方向与预期相反（Δ 为负），脚本会提示检查舵机接线/指令符号

### 实机测试注意事项

- **中立参考**：参考四元数在第一个 mocap 帧捕获。启动采集前球杆必须处于中立位，否则所有姿态带偏移。
- **IMU 自清零**：连接后 ~2.5s 自动清零，发生在 STATIC 段，期间保持静止。
- **轨迹时长**：`exploration_duration_s` 必须 ≥ 轨迹总时长。
