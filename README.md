# 缆驱动球关节研究平台

2-DOF 四缆绳驱动球杆系统（球杆绕球心两自由度摆动：**前后 / 左右**，期望工作范围 ≤65°）。实验平台：多传感器同步采集、最小闭环控制、数据驱动前馈建模。

> 项目正在从最小实现（`example_code/steering_motor`）出发重构。旧实现已归档至 `archive/`，顶层设计见 **[docs/top_level_design.md](docs/top_level_design.md)**。

## 硬件

| 设备 | 用途 | 默认接口 |
| --- | --- | --- |
| Nokov 光学动捕 | 球杆姿态反馈（真值） | 服务器 `10.1.1.198` |
| IM948 BLE IMU | 惯性姿态 / 角速度 | BLE `A5:B2:90:FF:4A:12` |
| 六维力传感器 | 缆绳张力 | 串口 COM6，MODBUS-RTU |
| 4× 舵机 (STM32F103) | 缆绳驱动 | 串口 COM5，115200 baud |

舵机协议：`0xAA | id(1B) | int16_be(2B) | 0x55`，共 5 字节。

- `id=1` → 前后对；`id=2` → 左右对；`id=0` → 放松（急停）。
- 耦合在**固件内**完成（一端 `+offset`、另一端 `-offset`），主机只发 2 路差分 offset。

## 目录结构

```
spherical_joint/
├── hardware/    L0 硬件/协议层（协议帧、舵机 offset、动捕读帧）
├── collect/     L1 采集层（统一 schema、多传感器、分段）
├── excite/      L2 激励层（开环差分激励 + 安全监护）
├── control/     L3 控制层（最小闭环 PID）
├── model/       L4 建模层（前向 F / 残差分解 / 前馈）
├── integrate/   L5 集成层（u_ff + u_fb，A/B）
├── archive/     归档的旧实现（参考，勿在新代码中 import）
└── docs/        设计文档
```

## 安装

```bash
conda activate spherical_joint
pip install -r requirements.txt
```

Nokov 动捕 SDK 非 pip 包，需单独安装厂商 SDK（安装后 `from nokov import nokovsdk` 可用）。

## 设计文档

| 文档 | 内容 |
| --- | --- |
| [docs/top_level_design.md](docs/top_level_design.md) | 顶层重构设计（架构 / SSoT / 方法论 / 里程碑） |
