# feedforward —— 最小前向建模流水线

本阶段目标：**先跑通"数据 → 前向模型 → rollout 评估"整条链路，观察结果**，模型侧暂时不考虑死区/静摩擦。

## 思路

- 无 IK 开环激励：直接在舵机**差分/共模**空间 `(d1, d2, p)` 设计大摆幅、低速、平滑轨迹（见 `excitation.py`）。
- 前向模型 `F`：MLP 学 `(q 历史, u 历史) → Δq`（下一拍姿态增量），ground truth 是动捕解算的 `current_pitch/current_yaw`。
- 训练/评估：训练集做 leave-one-trajectory-out，留出轨迹做 1-step 与 free-running rollout 评估。

## 数据格式

第一版只读 `servo_data.csv`，它已经同时包含：

- 输入 `u`：`servo_1_target_deg .. servo_4_target_deg`（**归一化 [-1,1]**，列名带 deg 但存的是归一化值）
- 输出真值 `q`：`current_pitch`, `current_yaw`（动捕解算姿态，度）

IMU / 张力后续作为额外特征并入，本版不涉及。

## 运行

```bash
conda activate spherical_joint

# 离线合成数据跑通全流程
python -m feedforward.train --synthetic --epochs 300

# 真实采集数据（servo_data.csv 所在目录）
python -m feedforward.train --data-dir <session_dir> --epochs 300
```

输出在 `feedforward/outputs/`：`metrics.json`（误差指标）、`rollout.npz`（真值/预测序列）、`rollout.png`（对比图）。

## 观察点（拿到结果后看什么）

1. **1-step RMSE**：模型能否解释单拍增量（含噪声地板）。
2. **rollout 误差是否随 rollout 步数发散**：前向模型 free-running 的漂移程度。
3. **误差集中在哪**：若集中在低速/换向处，说明死区/静摩擦是主要残差（本版模型未建模），即后续加摩擦结构的依据。
