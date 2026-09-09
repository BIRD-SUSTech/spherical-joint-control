# 稳态前馈数据驱动迭代范式（Playbook）

> 目的：① 记录本平台稳态前馈 v1→v7 的收敛历程（数据留痕）；② 抽象成可复现 SOP，
> 硬件更新后照单高效重跑。

## 〇、rig2 重启（新硬件，从头再跑一轮）

- **硬件变更**：缆绳弹性更小、舵机更强、结构不变；固件协议 v2 单帧多通道（11 字节）。
- **接口不变**：舵机 COM5 / 力 COM3 / 动捕 10.1.1.198；力只取 ch1–ch4。
- **旧数据全部归档作废**：`configs/controller_v1..v8`、`calibrations/rig1.json`（rig1 旧硬件）。
- **新命名**：`calibrations/rig2.json`（标定）、`configs/rig2_v*.json`（前馈）。
- **安全铁律**：新舵机更强 → 真实增益更大 → 用旧增益做开环会超调。开环前必须先 M3 标定，
  `rig2.json` 增益为 null 时 orchestrator 拒绝开环（已实现护栏）。
- **本轮目标**：**稳态前馈**（g(q) + gain_cross + 收敛段精化 + 大角度滚雪球）。动态前馈
  （迟滞/摩擦/速度）**暂缓**——待稳态前馈 A/B 收敛后再讨论。
- 命令清单：`current_stage_cmds.txt`（已重写为 rig2 从头流程，只到稳态前馈）。

---

## 一、阶段总结（本平台数据留痕）

每轮一个 A/B 循环，判据固定：**每级只进不退，否则回退**。

| 版本 | 形态 | A/B 结果 | 结论 | commit | 数据位置 |
|---|---|---|---|---|---|
| v1 | 方向分段增益 | 5° −24%，10° fb −10% | 有效但大角度欠拟合 | 55318cb | M6_verification |
| v3 | 开环三阶逆映射 g(q)+bias+slew | 连续轨迹 ±12° −20~−43% | merge | 3651c10 | M7_verification |
| v4 | + 迟滞 h·sign(q̇) | fb +2.7%（噪声内） | **否决**（诊断：换向残差实为增益高估） | e3ba1fb | M8_verification |
| v5 | 收敛段拟合（全量） | fb −3%、lr +6.7% | **否决**（lr 被收敛段带偏） | 36f41bb | M9_verification |
| v6 | 分轴：fb 收敛段 + lr 开环 | fb 平均 −11%、lr≈0 | merge | cb65ad6 | M10_verification |
| v7 开环 | 大角度开环重拟合 | circle20 从近失稳修复 | 过渡版（bias 退回不准） | cb062f3 | M11/M12_verification |
| v7 真版 | 收敛段 ±20° + fade | 稳态/大角度/waypoints 全过 | **merge（最终）** | cf3a05a/e79dca4 | M13/M14/M15_verification |
| v8 候选 | + 几何耦合交叉项（加性） | 留出验证 fb/lr RMSE −40%/−41% | **待实机 A/B**（离线护栏过） | （未上机） | M14/M15 收敛段 |

### 关键数据留痕

| 数据 | 值 | 来源 |
|---|---|---|
| 方向不对称 | fb pos/neg = 0.072/0.043（1.67×） | M6 |
| 幅值爬升 | 增益 ±5° 0.057 vs 小幅 0.0357（1.6×） | M4 |
| bias（h(0)） | fb +0.76°、lr −0.025° | M5 开环拟合 |
| 收敛段 bias | fb −20.7（准）、lr +1.06（v3 附近） | M10/M13 |
| 大角度折叠 | v6 在 +15~20° g(q) 反转（b3=−0.032 过陡） | M11 |
| 纯 PID 基线 | circle20 fb MAE 1.19 | M15 |
| 最终 v7 真版 | circle20 fb −13~−15%、waypoints lr −33~−38% | M14/M15 |
| 几何耦合（加性交叉） | 留出 RMSE：v7 39.9/31.4 → v8 24.0/18.5（offset） | M14/M15 收敛段 |

### 每轮的方法论结论（可迁移）

1. **换向残差 ≠ 迟滞**：先按 q_d 分箱，区分"残差∝q_d（增益高估）"vs"残差∝sign(q̇)（迟滞）"，再决定加什么项。
2. **收敛段拟合要分轴决策**：只修"开环拟合被动态污染"的轴（fb），不动"开环已准"的轴（lr）。
3. **错误前馈是负资产**：外推折叠的前馈（v3 在 20°）比纯 PID 还差；不折叠的 v7 才达全局最优。
4. **三阶项大角度外推易折叠**：覆盖大角度后重拟合，b3 自然趋小。

---

## 二、可复现 SOP（硬件更新后照单执行）

### 前置（一次性，非前馈特有）

```
M0 归档旧实现 → M1 L0+L3 最小闭环 → M2 L1 采集 → M3 标定+baseline → M4 开环激励
```

产出：`calibrations/rig1.json`（符号+bias）、`configs/`（控制器参数）、baseline 数字。

> **M3 内含一步共模预紧标定（§8.3.2，需固件共模通道）**：用力传感器/换向死区标定
> `c_fb(q_lr)`、`c_lr(q_fb)`（每对两缆同收、只紧不松的偶函数），在 M4 开环辨识**之前**
> 把系统预紧到"四缆张力 ≥ T_min"。预紧改变刚度/摩擦 → M4 起的 h(u)/g(q)/PID 全部要在
> 已预紧系统上重做。

#### 共模预紧标定 SOP（M3，前置，~半天）

1. 固件实现协议 v2 单帧多通道（`hardware/protocol.py`：`0xAA|CMD|CH1..CH4|0x55`，
   11 字节；CMD=0x01 把 4 个 int16 一次性原子写入 4 舵机，CMD=0x00 放松）。
2. 开环扫 2D 工作区，`collect/sensors/force.py` 读每根缆张力 `T_i(q_fb,q_lr)`（或
   triangle/eight 换向处测死区宽度 `δ_axis(q_orth)`）。
3. 拟合偶函数 `c_axis(q_orth) = c0 + c2·q_orth² + …`（或按 |q_orth| 分箱查表），判据
   `c_axis = argmin c  s.t. min_i T_i ≥ T_min`（或 `δ + margin`）。
4. 写入 `calibrations/rigN.json`（与控制器参数严格分离；它属标定，不属 configs/）。
5. 预紧生效后重跑 M4 开环辨识 + Kp/Kd 扫描（刚度已变，旧参数作废）。

### 级 1：方向分段增益（~1 天）

1. 开环数据 `fit_direction_gains` → 每轴正负方向增益。
2. 生成 `controller_v1.json`（direction_gains）。
3. A/B：`--ab --circle 5 6 --controller-config controller_v1.json`。
4. 判据：5° 改善 → merge；10° 若 fb 欠拟合 → 进级 1.5。

### 级 1.5：参数化逆映射 g(q)（~1 天）

1. 开环数据 `model.fit_controller --session <开环会话>` → 三阶逆映射系数。
2. 生成 `controller_v2.json`（gain_poly + slew_limit）。
3. A/B：连续轨迹（circle/lissajous/eight/variable-circle）。
4. 判据：连续轨迹 −20%+ → merge。

### 级 1.6：几何耦合解耦（加性交叉项，~1 天）

前提：先有 ≥2 个闭环收敛段会话（覆盖 2D 工作区，如 circle20）。

1. `model.fit_controller --converged-only --couple --session <会话1> <会话2> …`
   联合拟合 `u = own(q_self) + cross(q_other)`，并做留一会话交叉验证。
2. **护栏**：只有 `交叉 RMSE ≤ 1D RMSE`（fb 与 lr 两轴都满足）才写 `gain_cross`，
   否则自动回退 1D（输出文件只有 `gain_poly`）。
3. 产物 `configs/controller_vN_candidate.json`（含 `gain_poly` + `gain_cross`）。
4. A/B：`--ab --controller-config <vN_candidate> --baseline-controller-config controller_v7.json`
   （同轨迹、同中立起步、段间回正）。
5. 判据：稳态 MAE/RMSE 两轴不劣于 v7 才 merge；任何一轴回退 → 否决，只保留 v7。

> 关键：几何运动学只决定"交叉项方向"（fb 倾斜影响 lr 平衡，反之亦然），系数全由
> 数据拟合。不要用几何算系数（实机缆路径/摩擦与理想球面大圆弧可能差很远）。

### 级 2 诊断（可选，先诊断再决定）

1. 用前馈后残差，**按 q_d 分箱**区分"增益高估 vs 迟滞"。
2. 增益高估 → 进机制 A；迟滞 → 才加 h·sign(q̇)。

### 机制 A：收敛段精化（关键一步）

1. 闭环采集（前馈段，多轨迹多工作点）。
2. `model.fit_controller --converged-only --session <闭环会话>` → 收敛段 u_log 拟合。
3. **分轴决策**：看 lr bias 是否偏离开环值，偏离则 lr 退回开环（只 fb 用收敛段）。
4. A/B：`--baseline-controller-config <上一版>`。

### 滚雪球大角度（扩工作范围）

1. `--open-loop --large-angle` 采集大角度开环（±15°/±20°）。
2. 重拟合 g(q) 覆盖大角度（修外推折叠）。
3. 大角度闭环 A/B 采集（circle/waypoints ±15/20°）。
4. `fit_controller --converged-only`（合并大角度收敛段）。
5. A/B 验证。

### 启动瞬态（如出现）

1. 诊断启动区峰值（0–2s）。
2. `--startup-fade 1.0` 前馈渐入。
3. A/B：启动峰值 ≤ baseline。

---

## 三、关键工具与判据

| 工具 | 用途 |
|---|---|
| `model.fit_controller` | 拟合 g(q)（开环 / 收敛段）；`--couple` 拟合几何耦合交叉项（留出护栏） |
| `collect.orchestrator --ab` | 自动 A/B（baseline 可配 --baseline-controller-config） |
| `valuation.evaluate --all` | 分段指标 + 6 子图 + 分段图/gif |
| `--startup-fade` / `--inter-segment-settle` | 启动渐入 / 段间回正（A/B 公平性） |

**判据铁律**：
1. A/B 每级只进不退（持平或更好才 merge）；
2. 稳态口径（t>5s）为主，全段口径参考（启动瞬态污染）；
3. session_metadata.json 自动记录 config（轨迹/标定/前馈系数），数据可复现；
4. 每轮数据归档 NAS `M*_verification/data/`，报告留 commit hash。
