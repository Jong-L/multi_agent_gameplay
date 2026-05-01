# GRALP 避障机制分析与对当前项目的启发

> 分析对象：`D:/schoolTour/github repos/GRALP`（Generalized-depth Ray-Attention Local Planner）
> 对比项目：`d:/schoolTour/softwares/multi-agent-gameplay`（Tiny Swords 多智能体竞技场）

---

## 一、GRALP 项目概述

GRALP 是一个**无地图（Map-Free）GPU 批量强化学习局部规划器**，在完全随机化的虚拟障碍环境中训练 PPO 策略。它的核心理念是：**不需要真实的地图/几何体，而是在每步随机生成虚拟的"光线视野"（FOV），让智能体学会纯粹基于距离传感器数据避障和导航。**

项目结构：

```
GRALP/
├── config/
│   ├── env_config.json          # 环境/观测/奖励配置
│   └── train_config.json        # PPO 超参配置
├── env/
│   ├── sim_gpu_env.py           # 核心：批量随机化光线环境
│   └── ray.py                   # 射线工具（射线数推导、射线扫描）
├── rl_ppo/
│   ├── encoder.py               # RayEncoder 主干网络（注意力+卷积）
│   ├── ppo_models.py            # 策略/价值网络
│   └── ppo_train.py             # 训练入口
└── tools/
    └── setup_api.py             # ONNX 模型导出
```

---

## 二、GRALP 避障机制详解

### 2.1 核心设计：虚拟化光线 FOV（No Map, Only Rays）

GRALP 的避障机制**完全不依赖真实地图几何体**。它的核心逻辑是：

```
每步重新随机生成一个虚拟的"光线视野" → 
智能体认为这些光线距离就是真实障碍 → 
策略学习根据光线数据避障导航
```

这意味着环境中**没有静态墙壁、没有碰撞体、没有 TileMap**，只有一个虚拟的 360° 距离传感器读数。

### 2.2 光线生成算法（`_resample_fov_and_ref()`）

```python
# 1. 确定每条射线是"空"还是"障碍"
blank_ratio = Gaussian(base=40%, sigma=jitter*std)  # 空射线比例
mask_empty = random(R) < blank_ratio                # R条射线各自的空/障掩码

# 2. 障碍射线的距离采样
if narrow_passage_gaussian:  # 窄通道模式
    dist = abs(N(0, σ=patch_meters * std_ratio))  # 半高斯，障碍更近
else:                        # 均匀模式
    dist = uniform(0, view_radius)

# 3. 填充距离
rays_m = where(mask_empty, view_radius, dist)  # 空射线=最大距离
```

**关键参数**（来自 `config/env_config.json`）：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `patch_meters` | 10.0 | 视野半径（米）|
| `ray_max_gap` | 0.6 | 视野边界上相邻射线最大弧长间隔 → 决定射线数 |
| `blank_ratio_base` | 0% | 基准空射线比例 |
| `blank_ratio_randmax` | 50% | 额外随机空射线范围 |
| `narrow_passage_gaussian` | true | 是否开启窄通道模式 |
| `narrow_passage_std_ratio` | 0.15 | 窄通道半高斯标准差系数 |

由此推导射线数：**R = ⌈2π × patch_meters / ray_max_gap⌉ = ⌈2π × 10 / 0.6⌉ = 105 条射线**

### 2.3 碰撞判定（`step()` 中的碰撞检测）

GRALP 的碰撞判定是**纯数学的**，无需物理引擎：

```python
travel = hypot(dx_local, dy_local)          # 本步实际移动距离
angle  = atan2(dy_local, dx_local)          # 移动方向
ray_d  = interp_ray_distance(rays_m, angle) # 沿移动方向的射线距离（线性插值）
collided = (travel > ray_d) & (ray_d > 0)   # 超出射线距离 = 碰撞
```

**核心思想**：智能体在当前前进方向上的"许可距离"由射线距离决定。如果移动距离超过该射线指示的障碍距离，就判定为碰撞。

**碰撞惩罚**：
```python
reward -= w_collision * (1 + |v_world| / vx_max)  # 速度越快惩罚越重
```

### 2.4 任务点导航：LOS 裁剪

任务点（目标导航点）的生成也是基于光线的：

```python
# 1. 只选"安全"方向（射线距离 ≥ safe_distance）
safe = rays_m >= safe_distance_m
pick = random_choice(where(safe))

# 2. 全局任务点被"投影"到 LOS 可见范围内
local_task = min(global_task_distance, ray_distance_in_that_direction)
```

这确保了：智能体永远不会被告知去一个"被障碍物挡住"的位置。

### 2.5 观测空间设计

```python
observation = [rays_norm(R),      # R条归一化射线距离 (0~1)
               sin_ref, cos_ref,  # 目标方向（机体坐标系）
               prev_vx/lim,       # 上一帧线速度
               prev_omega/lim,    # 上一帧角速度
               Δvx/(2*lim),       # 速度变化
               Δomega/(2*omega_max), # 角速度变化
               dist_to_task/radius]  # 到任务点距离

# 总维度 = R + 7
```

**核心设计原则**：
- 没有绝对位置（智能体不知道"自己在哪里"）
- 没有全局地图（只知道当前的光线读数）
- 包含运动学历史（速度、加速度）
- 目标以相对方向编码（sin_ref, cos_ref 而非绝对坐标）

### 2.6 RayEncoder 网络架构

```
输入: [B, R+7]  观测向量
         ↓
    ┌─────────────────────┐
    │  RayBranch (光线分支) │  1D深度可分离卷积 + GELU + Squeeze-Excite
    │  4层，膨胀率 [1,2,4,8] │  用圆周padding保持旋转不变性
    └────────┬────────────┘
             ↓ [B, hidden, R]
    ┌── to_k ──┐  ┌── to_v ──┐          ┌ pose_mlp ─┐
    │ K [R,D]  │  │ V [R,D]  │          │ pos_bias  │
    └────┬─────┘  └────┬─────┘          └─────┬─────┘
         │              │                      │
         │    Multi-Query Attention            │
         │    Q = learnable_query + pos_bias   │
         │    z = Softmax(QK^T/√D) · V        │
         │              ↓                       │
         │    z_mean, q_mean, gavg(V)          │
         └──────────────┬──────────────────────┘
                        ↓
              Fusion Head (2层MLP → 256d)
                        ↓
          ┌─────────────┴─────────────┐
    策略头: Linear(256→3)       价值头: Linear(256→1)
    tanh压缩高斯分布
```

**网络特点**：
1. **1D 深度可分离卷积**：处理光线序列的局部邻域关系（相邻方向的光线更相关）
2. **Squeeze-Excite 块**：自适应标定通道重要性
3. **多查询多头注意力**：用可学习查询向量 + 姿态偏置，从光线特征中提取关键信息（如"最近的障碍在哪个方向"、"哪个方向有空隙"）
4. **圆周padding**：光线是环形的，左边和右边在物理上是相邻的

### 2.7 动作空间

连续动作 `[vx, vy, omega]`（或仅2轴 `[vx, omega]`，vy=0）：
- `vx`：前进线速度 ∈ [-0.6, 0.6] m/s
- `omega`：角速度 ∈ [-1.5, 1.5] rad/s
- 使用 tanh 压缩高斯分布输出

---

## 三、当前项目避障机制回顾

### 3.1 物理射线检测

```gdscript
# play_scene.gd _build_map_state()
func _build_map_state(player: Player) -> Array[float]:
    var collision_mask := 4  # CollisonDecoration 碰撞层
    for dir in ray_directions:  # 32条均匀分布射线
        var end_pos = player_pos + dir * max_distance
        var query = PhysicsRayQueryParameters2D.create(player_pos, end_pos, collision_mask)
        var result = space_state.intersect_ray(query)
        if result.is_empty():
            map_state.append(1.0)  # 无碰撞 = 最远
        else:
            map_state.append(collision_distance / max_distance)  # 归一化
```

对比维度：

| 维度 | GRALP | 当前项目 |
|------|-------|----------|
| 障碍来源 | 虚拟随机生成（每步重采样） | 真实物理碰撞体（CollisonDecoration tiles） |
| 射线数 | 105（可配置） | 32（可配置，game_config.ray_count） |
| 射线分布 | 均匀 360° | 均匀 360° |
| 归一化 | distance / view_radius | collision_distance / max_distance |
| 障碍持久性 | 每步重新随机 | 静态地图，永久不变 |
| 碰撞检测 | 数学插值（travel vs ray_dist） | Godot 物理引擎碰撞事件 |
| 检测层 | 虚拟（无实际物理层） | collision_mask=4 |
| 碰撞惩罚 | -w × (1+v_ratio) × collided | -wall_collision_penalty |

### 3.2 观测空间中的地图信息

```
当前观测 = [self_state (4d), nearby_players (15d), nearby_balls (32d),
            nearby_enemies (25d), map_state (32d)]
总维度 = 108
```

`map_state` 作为 32 维向量被展平拼接，**没有空间结构保留**。

### 3.3 墙避障奖励塑形

当前项目的 wall potential shaping 使用势能函数：
- `LINEAR`: Φ = (penalty/radius) × d_min - penalty
- `INVERSE`: Φ = -1/(d_min + 1/penalty)
- `COLLISION`: 仅碰撞时惩罚（无塑形）

塑形公式：`F = γ × Φ(s_t) - Φ(s_{t-1})`（Ng et al. 1999 势能塑形）

---

## 四、GRALP 对当前项目的核心启发

### 4.1 ⭐ 启发一：用注意力机制替代扁平的 map_state

**当前问题**：32 维 `map_state` 被当作扁平的浮点数列表输入给 MLP，完全丢失了光线的**空间邻域关系**和**相对方向信息**。

**GRALP 方案**：RayEncoder 通过 1D 卷积 + 多查询注意力，让网络能够：
1. 学习局部邻域模式（"连续几条射线都很近 = 面前有一堵墙"）
2. 通过注意力自动聚焦关键方向（"哪个方向有出口"）
3. 保持旋转等变性（圆周padding）

**实施建议**：

```python
# Python 端添加一个轻量级的 RayAttention 模块
class LightRayAttention(nn.Module):
    def __init__(self, n_rays=32, hidden=32, d_model=64):
        # 1D Conv + Attention over rays
        self.conv = nn.Conv1d(1, hidden, 7, padding=3, padding_mode='circular')
        self.attn = nn.MultiheadAttention(d_model, num_heads=4)
    
    def forward(self, map_state):  # [B, 32]
        x = self.conv(map_state.unsqueeze(1))  # [B, hidden, 32]
        x = x.permute(2, 0, 1)  # [32, B, hidden]
        out, weights = self.attn(x, x, x)
        return out.mean(0)  # [B, d_model]
```

**工作量**：中等。需要在 Python RL 端修改网络结构，但观测维度不变（只是处理方式改变）。

### 4.2 ⭐⭐ 启发二：障碍密度课程学习（Blank Ratio Curriculum）

**GRALP 做法**：
- 训练开始：`blank_ratio_base=40%` — 40%~90% 射线为空，环境稀疏
- 训练后期：`blank_ratio_base=0%` — 仅 0%~50% 射线为空，环境密集
- 逐步降低 blank ratio，让智能体从"开阔环境"逐步适应"密集障碍"

**对当前项目的启发**：当前地图中的 CollisonDecoration 是静态的，但我们可以在物理模拟层面实现类似效果：

**方案A（代码层面 - 简单）**：训练时动态启用/禁用部分装饰物碰撞体
```gdscript
# 在 game_config 中添加
@export var obstacle_density: float = 1.0  # 0.0~1.0
# 按比例随机禁用 collision_decoration tiles 的碰撞
```

**方案B（观测层面 - 最简单）**：在 `_build_map_state()` 中人为增加"假空射线"
```gdscript
# 每帧随机将某些碰撞距离改为 1.0（假装没有障碍）
if randf() < blank_ratio:
    map_state[i] = 1.0  # 该射线假装没看到障碍
```

这样无需修改地图，只需在观测生成时注入噪声，逐步降低 blank_ratio 即可实现课程学习。

**工作量**：低。只需修改 `_build_map_state()` 函数。

### 4.3 ⭐⭐⭐ 启发三：碰撞判定模型改进

**当前问题**：碰撞检测完全依赖 Godot 物理引擎的碰撞事件（`player.last_collison_data`），存在以下局限：
1. 碰撞发生后才检测 → 反应延迟
2. 无法预测即将发生的碰撞
3. 惩罚发生在碰撞物理帧，而非碰撞风险帧

**GRALP 做法**：在每步动作执行时，**主动计算**移动方向上的许可距离：

```python
travel = vx * dt           # 本步移动距离
ray_d = interp_ray(angle)  # 该方向的障碍距离
if travel > ray_d:         # 超过 = 碰撞
    collision!
```

**实施建议**：在 `_physics_process` 或 `reward_manager` 中增加**预测性碰撞检测**：

```gdscript
# 在玩家移动前，预测是否会撞墙
var move_dir := player.velocity.normalized()
var move_angle := move_dir.angle()
var ray_idx := _angle_to_ray_index(move_angle)
var collision_distance_norm := map_state[ray_idx]
var move_distance := player.velocity.length() * delta

if collision_distance_norm < 1.0:  # 该方向有障碍
    var actual_dist := collision_distance_norm * vision_radius
    if move_distance > actual_dist:
        # 预测碰撞！在碰撞发生前就给惩罚
        add_predictive_collision_penalty(player, actual_dist / move_distance)
```

**优势**：
- 碰撞前预警（而非碰撞后惩罚）
- 距离越近惩罚越重（更精细的梯度信号）
- 不依赖物理引擎的碰撞事件

**工作量**：中等。需修改 `reward_manager._process_wall_collision()`。

### 4.4 ⭐ 启发四：观测空间增强 — 更丰富的运动学历史

**当前观测**（运动学部分仅 flip_h）：
```
self_state = [pos_x, pos_y, hp_ratio, flip_h]
```

**GRALP 观测**（包含 7 维运动学上下文）：
```
[rays_norm(R), sin_ref, cos_ref, prev_vx/lim, prev_omega/lim, 
 Δvx/(2*lim), Δomega/(2*omega_max), dist_to_task]
```

**关键差异**：
1. 上一个时间步的速度信息（`prev_vx`, `prev_omega`）
2. 加速度信息（`Δvx`, `Δomega`）→ 帮助推断运动趋势
3. 到目标的方向编码（`sin_ref, cos_ref`）而非绝对坐标

**实施建议**：在观测中添加**速度历史**和**目标方向**信息：

```gdscript
# 在 vision_sensor.scan() 的 self_state 中添加
var self_state = [
    player_pos.x / half_arena,
    player_pos.y / half_arena,
    player.current_health / player.max_health,
    int(player.animated_sprite.flip_h),
    # 新增运动学信息
    player.velocity.x / max_speed,   # 归一化速度
    player.velocity.y / max_speed,
    # 若有目标方向，可添加 sin_angle, cos_angle
]
```

**注意**：这会改变观测维度，需要同步更新 `controller.get_obs_space()` 和 Python 端的网络输入尺寸。

**工作量**：低。主要修改 `vision_sensor.gd` 和 `controller.gd`。

### 4.5 ⭐ 启发五：可选 — 窄通道专项训练

**GRALP 做法**：`narrow_passage_gaussian=true` 时，障碍物距离服从半高斯分布（标准差 = 视野半径 × 0.15），大量障碍集中在近处，模拟狭窄通道场景。

**对当前项目的启发**：可以在当前竞技场中生成临时性的"通道"或"隧道"：

**方案A**：动态生成碰撞体
- 使用程序化生成的 `StaticBody2D + CollisionShape2D` 临时添加障碍
- 训练时随机切换通道形状和位置
- 训练完成后移除

**方案B**：调整装饰物密度
- 利用现有 CollisonDecoration tiles，在训练时动态调整 collisions 的启用比例
- 密集模式 = 窄通道训练

**工作量**：中等（方案A）到低（方案B）。

### 4.6 ⭐ 启发六：连续动作空间的长期考量

GRALP 使用连续动作 `[vx, omega]` + tanh 压缩高斯策略。当前项目使用 6 个离散动作（上/下/左/右/攻击/待机）。

**避障效果差异**：
- 离散动作导致"之字形"移动，在狭窄通道中容易卡住
- 连续动作可以产生平滑轨迹，更适合避障

**建议**：暂时保持离散动作（因为当前项目是竞技场对战，动作设计与战斗绑定），但在未来如果需要做"纯避障导航"子任务，可以考虑添加连续动作头。

---

## 五、实施优先级建议

| 优先级 | 启发项 | 预期收益 | 工作量 | 风险 |
|--------|--------|----------|--------|------|
| **P0** | 4.3 预测性碰撞检测 | 高：让碰撞信号更及时 | 中 | 低 |
| **P0** | 4.2 观测层面 blank ratio 课程学习 | 高：渐进式训练 | 低 | 极低 |
| **P1** | 4.4 运动学历史观测增强 | 中高：更好的状态表示 | 低 | 中（需改观测维度） |
| **P1** | 4.1 注意力编码 map_state | 中：更好的空间理解 | 中 | 中（网络结构变更） |
| **P2** | 4.5 窄通道专项训练 | 中：特定场景能力 | 中 | 低 |
| **P2** | 4.6 连续动作空间 | 高但远：根本性改变 | 高 | 高 |

---

## 六、关键代码对比

### 6.1 碰撞检测

| | GRALP | 当前项目 |
|---|---|---|
| **文件** | `env/sim_gpu_env.py:197-200` | `reward_manager.gd:669-673` |
| **机制** | 数学插值预测 | 物理引擎事件 |
| **时机** | 动作执行时 | 碰撞发生后 |
| **公式** | `travel > interp_ray(angle)` | `collider is TileMapLayer` |

### 6.2 光线观测

| | GRALP | 当前项目 |
|---|---|---|
| **文件** | `env/sim_gpu_env.py:336-363` | `play_scene.gd:445-481` |
| **方法** | 虚拟随机采样 | PhysicsRayQueryParameters2D |
| **射线数** | 105（2πR/gap） | 32（game_config） |
| **障碍** | 每步随机 | 静态 CollisonDecoration |
| **编码** | 卷积+注意力 → 256d | 展平拼接 → 32d |

### 6.3 网络处理光线

| | GRALP | 当前项目 |
|---|---|---|
| **文件** | `rl_ppo/encoder.py:74-162` | Python 端 MLP（需确认） |
| **架构** | RayBranch (dw-conv+SE) + MHA | 全连接线性层 |
| **空间结构** | 保留（circular conv） | 丢失（flat concat） |
| **注意力** | 多查询多头注意力 | 无 |

---

## 七、总结

GRALP 的核心贡献是证明了：**在无地图、纯光线距离传感器数据下，通过注意力机制 + 课程学习，可以训练出有效的避障导航策略。** 对当前项目最有价值的三个启发是：

1. **预测性碰撞检测**（启发三）：在碰撞发生前就给惩罚信号，而不是等物理引擎报碰撞
2. **Blank ratio 课程学习**（启发二）：通过在观测层面注入"假空射线"，渐进式增加障碍密度，降低训练难度
3. **运动学历史**（启发四）：在观测中加入速度/加速度信息，让智能体"感知"自己的运动趋势

这三个改进的实现成本低、风险小，可以在不改变游戏核心机制（离散动作、对战玩法）的前提下，显著提升智能体的避障能力。
