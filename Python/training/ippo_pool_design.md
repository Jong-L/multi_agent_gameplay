# custom_ippo_pool.py 设计文档

## 目录

1. [问题背景与实验目的](#1-问题背景与实验目的)
2. [总体架构：三阶段流水线](#2-总体架构三阶段流水线)
3. [Phase A: IPPO Bootstrap —— 产生对手池原料](#3-phase-a-ippo-bootstrap--产生对手池原料)
4. [Phase B: Opponent-Pool Cycle —— 对抗进化核心](#4-phase-b-opponent-pool-cycle--对抗进化核心)
5. [Phase C: Evaluation —— 对比验证](#5-phase-c-evaluation--对比验证)
6. [核心数据结构](#6-核心数据结构)
7. [代码模块映射](#7-代码模块映射)
8. [配置参数速查](#8-配置参数速查)

---

## 1. 问题背景与实验目的

### 1.1 核心问题

多智能体强化学习中，**同时训练多个智能体（IPPO）会引入非平稳性**：每个智能体的策略在变，其他智能体也在变，导致每个智能体面对的环境是动态的。这容易使训练不稳定，策略陷入局部最优。

### 1.2 实验假设

> **经过对手池对抗博弈训练的智能体，比直接 IPPO 训练的智能体表现更稳定、泛化性更强。**

直观理解：IPPO 同时更新所有智能体，策略之间容易形成特定的"共适应"关系。而对手池训练让主智能体持续面对多样化的历史对手快照，迫使它学会鲁棒的策略，而非针对某个特定对手过拟合。

### 1.3 实验方案总览

```
Step 1 (外部)  单智能体 PPO 课程学习  →  ppo_agent_{0..3}.pt
Step 2 (Phase A)  IPPO Bootstrap        →  对手池原料 + direct_ippo_agent0
Step 3 (Phase B)  Opponent-Pool Cycle   →  pool_agent0
Step 4 (Phase C)  Evaluation            →  对比 CSV + 统计
```

---

## 2. 总体架构：三阶段流水线

### 2.1 设计原则

| 原则 | 体现 |
|---|---|
| **单一职责** | 每个 `run_mode` 只做一件事，通过文件系统传递状态 |
| **复用现有模块** | rollout/GAE/loss/logging 全部从 `custom_ippo.py` 导入，不重复实现 |
| **配置即实验** | 所有实验超参都是 dataclass 字段，CLI 可覆盖 |
| **可中断可恢复** | 每个阶段产出完整 checkpoint，下一阶段可独立运行 |

### 2.2 三条 run_mode

```
                    ┌──────────────┐
  ppo_model_paths → │ ippo_bootstrap │ → bootstrap 中断点 + direct_agent0
                    └──────┬───────┘
                           │ 文件系统
                    ┌──────▼───────┐
                    │  pool_cycle   │ → pool_agent0
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   evaluate    │ → CSV
                    └──────────────┘
```

`--run-mode full` 自动串联 bootstrap → pool_cycle。

---

## 3. Phase A: IPPO Bootstrap —— 产生对手池原料

### 3.1 目的

在多智能体环境下，用预训练的 PPO 单智能体模型起步，让 4 个 agent **同时训练一段时间**，产生一批**多样性策略快照**（中断点），作为后续对手池的初始化原料。

### 3.2 为什么要两阶段

| 子阶段 | 设置 | 产出 | 目的 |
|---|---|---|---|
| **Phase A1** | save_checkpoint=True | 每 N episode 保存中断点 | 提供**策略多样性**：训练早期策略变化大，快照差异显著 |
| **Phase A2** | save_checkpoint=False | 仅保存最终模型（agent0） | 在 Phase A1 基础上继续收敛，产出 baseline 对照模型 |

### 3.3 关键设计决策

- **所有 agent 同时训练（train=True for all）**：快速产生互相适应的策略快照
- **从 PPO checkpoint 加载**：避免随机初始化的低质量策略浪费训练时间
- **仅加载 PPO 权重，不加载优化器/归一化器状态**：单智能体 PPO 的优化器动量对多智能体场景无意义
- **`total_timesteps` 使用累计值**：Phase A2 设为 `1M + 8M = 9M`，利用 `resume_from` 跳过已完成的更新

### 3.4 文件产出

```
saved-models/
├── ippo_bootstrap/
│   ├── ippo_bootstrap_episode10_agent0.pt    ← 对手池原料
│   ├── ippo_bootstrap_episode10_agent1.pt
│   ├── ippo_bootstrap_episode10_agent2.pt
│   ├── ippo_bootstrap_episode10_agent3.pt
│   ├── ippo_bootstrap_episode20_agent0.pt
│   └── ...                                   ← 共 ~10 组
└── ippo_direct/
    └── ippo_direct_agent0.pt                 ← baseline 对照模型
```

---

## 4. Phase B: Opponent-Pool Cycle —— 对抗进化核心

### 4.1 核心思想：简化版虚拟自我博弈（PFSP）

虚拟自我博弈（Fictitious Self-Play）的核心：**每轮训练一个主 agent 对抗历史对手策略的混合分布**，迫使主 agent 发展出能应对多样化对手的鲁棒策略。

本实现做了以下简化：
- 不是全连接博弈矩阵，而是每次选择对手组的 **槽位索引**
- 对手组内 agent 的快照时间步**对齐**（同一 slot_index）
- 难度衡量用**主 agent 回合平均奖励**代替 ELO 胜率

### 4.2 对手池结构

```
opponent_pool[4][20]

entries_by_agent:dict[int, list[PoolEntry]]
  0: [snap_0_0, snap_0_1, ..., snap_0_19]  ← agent 0 的快照队列
  1: [snap_1_0, snap_1_1, ..., snap_1_19]  ← agent 1 的快照队列
  2: [snap_2_0, snap_2_1, ..., snap_2_19]  ← agent 2 的快照队列
  3: [snap_3_0, snap_3_1, ..., snap_3_19]  ← agent 3 的快照队列
```

- 键 = **对手 agent 的身份**（0/1/2/3）
- 每个队列容量 20，新增快照入队尾，队列满时弹出队头（最旧）
- 初始用 Phase A 产出的最近 10 个 bootstrap 中断点填充

### 4.3 采样机制

#### 对手组采样

当主 agent = `k` 时，对手 id 集合 = `{0,1,2,3} \ {k}`：

```python
# 所有可用的对手组（按 slot 对齐）
candidates = [
    [entries_by_agent[opp_0][slot], entries_by_agent[opp_1][slot], entries_by_agent[opp_2][slot]]
    for slot in range(min(len per agent))
]
```

#### PFSP 采样分布

$$P(\text{slot}) \propto \exp\left(-\frac{\text{reward\_ema}(\text{slot}) - \min(\text{rewards})}{T}\right)$$

- `reward_ema(slot)`：主 agent 面对该对手组时的历史回合平均奖励 EMA
- 奖励**越低** → 概率**越高**（倾向于选择更困难的对手）
- `T`（temperature）：控制分布的尖锐程度，默认 0.5
- ε-greedy（默认 10%）：防止胜率估计过时

#### 对手切换与 RNN 状态

- 一次采样得到的对手组在**所有并行环境**中共享
- 当**任意**并行环境完成一个 episode → 记录本回合奖励 → 重新采样对手组
- 对手切换后，所有 agent 的 RNN 隐藏态**重置为零**（不同策略的 RNN 状态无意义）

### 4.4 训练流程

```
for round in 1..3:
    for main_agent_id in [0, 1, 2, 3]:
        train_pool_phase(main_agent_id, 800K steps):
            for each update:
                1. 采样对手组，加载对手权重
                2. 收集 rollout（仅主 agent 参与决策，对手用冻结策略）
                3. PPO 更新主 agent（对手不做梯度更新）
                4. 如果有 episode 完成 → 记录奖励 → 重采样对手
                5. 每 50K 步保存主 agent 快照入池

# 额外：agent 0 再训练 300 万步
train_pool_phase(agent_0, 3M steps)
```

### 4.5 设计决策

| 决策 | 理由 |
|---|---|
| 对手权重**冻结**（eval 模式，不更新） | 避免对手策略在训练中漂移，保持对抗目标的稳定性 |
| 对手组按 slot 对齐 | 保证同一组对手来自相同时期的训练阶段，策略水平接近 |
| 统一对手组用于所有并行环境 | 编码简化，避免逐环境采样 |
| RNN 状态在对手切换时重置 | 不同策略的隐藏态语义不兼容 |
| 队列式快照替换（FIFO） | 自然保持策略新鲜度，旧策略逐步淘汰 |

---

## 5. Phase C: Evaluation —— 对比验证

### 5.1 目的

在**相同的对照组**下，对比两个 agent 0 的表现：

- **direct_ippo_agent0**：Phase A 产出的 IPPO 直接训练模型
- **pool_agent0**：Phase B 产出的对手池博弈训练模型

### 5.2 评估流程

```
1. 从对手池中随机抽取 4 组对手 checkpoint
2. 对每组对手：
   - 加载 direct_ippo_agent0 + 3 个对手 → 跑 N 个 episode → 记录奖励
   - 加载 pool_agent0 + 3 个对手 → 跑 N 个 episode → 记录奖励
3. 输出 CSV：model_label × group_id × episode × agent_id × reward
4. 打印汇总：各模型 agent0 的 mean ± std
```

### 5.3 设计要点

- 两个模型面对**完全相同**的对手组（对照组对齐）
- 使用确定性动作（argmax），消除采样噪声
- CSV 输出规范化字段，方便后续 Pandas/绘图分析

---

## 6. 核心数据结构

### 6.1 `IppoPoolArgs`（配置层）

继承 `IppoArgs`，新增池相关参数。每种 `run_mode` 对应一组 CLI 参数，覆盖 dataclass 默认值。

### 6.2 `PoolEntry`（快照条目）

```python
@dataclass
class PoolEntry:
    checkpoint_path: str      # 权重文件路径
    agent_id: int             # 归属哪个 agent 槽位
    global_step: int          # 快照时的训练步数
    slot_index: int           # 在队列中的位置（由 OpponentPoolState 维护）
    source: str               # 来源标记（"initial" / round 名）
    main_reward_ema: float    # 主 agent 面对该快照所在对手组时的奖励 EMA
    n_games: int              # 参与的游戏局数
```

### 6.3 `OpponentPoolState`（池管理层）

```mermaid 架构图：
┌──────────────────────────────┐
│      OpponentPoolState       │
├──────────────────────────────┤
│ entries_by_agent → 快照仓库   │  按 agent_id 分桶的队列
│   {0: [...], 1: [...], ...}  │
├──────────────────────────────┤
│ stats → 难度追踪             │  按 (主agent, 对手组路径) 索引
│   {(0, group_a): stat, ...}  │
├──────────────────────────────┤
│ add_entry()    → 入队        │
│ sample_group() → PFSP 采样   │
│ record_result() → 更新统计   │
└──────────────────────────────┘
```

#### `entries_by_agent` vs `stats` 的分离设计

| | `entries_by_agent` | `stats` |
|---|---|---|
| 键含义 | 对手自身的 agent_id | `(主agent_id, 对手组路径)` |
| 用途 | 快照存储（中性、无偏见） | 难度追踪（按主 agent 独立） |
| 更新时机 | 保存快照时 | 回合结束时 |
| 采样参与 | 提供候选人列表 | 计算采样权重 |

**分离理由**：同一组对手快照（如 bootstrap_ep10 的快照组），主 agent 0 和主 agent 1 面对时的难度可能截然不同。`stats` 的键包含 `(main_agent_id, group)`，保证不同主 agent 的难度统计完全独立。

### 6.4 `TrainingContext`（训练状态层）

封装一次训练运行需要的所有可变状态（环境、agent、优化器、观测缓存、全局步数等），避免在函数间传递十几个参数。

---

## 7. 代码模块映射

### 7.1 文件依赖关系

```
custom_ippo_pool.py
├── 导入 from custom_ippo.py:
│   ├── IPPOAgent                    # 策略网络
│   ├── collect_parallel_rollout_ippo # rollout 采集
│   ├── train_agent_update           # PPO 单步更新
│   ├── log_ippo                     # 日志输出
│   ├── save_ippo_model              # 模型保存
│   ├── load_ppo_models_if_requested  # PPO 权重加载
│   ├── load_checkpoint_if_requested  # IPPO 中断点加载
│   └── 其他工具函数
├── 导入 from custom_ppo_dataclass.py:
│   ├── AgentConfig, IppoArgs, PoolEntry
├── 导入 from godot_env_wrapper.py:
│   ├── RewardNormalizer, init_training_setup, load_full_checkpoint
│
├── 本文件新增:
│   ├── OpponentPoolState  (对手池管理)
│   ├── TrainingContext     (训练状态封装)
│   ├── run_ippo_bootstrap()  → Phase A
│   ├── run_pool_cycle()      → Phase B
│   ├── run_evaluation()      → Phase C
│   └── main() 入口
```

### 7.2 关键函数与设计模式

| 函数 | 职责 | 设计模式 |
|---|---|---|
| `setup_training_context()` | 统一初始化入口，各模式共用 | Template Method |
| `run_ippo_training_job()` | 标准 IPPO 训练（Phase A 复用） | Strategy |
| `train_pool_phase()` | 对手池单 phase 训练（Phase B） | 独立实现（逻辑差异大） |
| `_sample_and_load_opponents()` | 对手采样 + 权重加载 | 分离关注点 |
| `_capture/restore_agent_states()` | 主 agent 状态快照/恢复 | Snapshot 模式 |
| `_with_train_flags()` | 动态设置 train/act_when_not_training | 配置转换 |

### 7.3 与 custom_ippo.py 的分工

| | `custom_ippo.py` | `custom_ippo_pool.py` |
|---|---|---|
| 训练哪些 agent | 全部（或按 `train` 标记） | 仅主 agent |
| 对手来源 | 当前实时策略 | 从对手池加载冻结快照 |
| 更新方式 | 逐 agent PPO 更新 | 仅主 agent PPO 更新 |
| 中断点保存 | 定时（按 episode） | 定时（按 step）+ 入池 |
| 对手切换 | 无 | 回合结束后 PFSP 重采样 |

---

## 8. 配置参数速查

### Phase A 相关

| 参数 | 含义 | 默认值 |
|---|---|---|
| `pool_bootstrap_checkpoint_timesteps` | Phase A1 训练步数 | 1,000,000 |
| `pool_bootstrap_extra_timesteps` | Phase A2 额外步数（累计） | 8,000,000 |
| `bootstrap_save_model_path` | 中断点保存路径 | `saved-models/ippo_bootstrap` |
| `bootstrap_final_save_model_path` | 最终模型保存路径 | `saved-models/ippo_direct` |
| `pool_final_save_agent_ids` | Phase A2 保存哪些 agent | `(0,)` |

### Phase B 相关

| 参数 | 含义 | 默认值 |
|---|---|---|
| `pool_slots_per_agent` | 每 agent 队列容量 | 20 |
| `pool_initial_keep_per_agent` | 初始从 bootstrap 载入数 | 10 |
| `pool_phase_timesteps` | 每 phase 训练步数 | 800,000 |
| `pool_rounds` | 轮回次数 | 3 |
| `pool_main_agent_order` | 轮回顺序 | (0,1,2,3) |
| `pool_final_agent_id` | 额外训练的主 agent | 0 |
| `pool_final_timesteps` | 额外训练步数 | 3,000,000 |
| `pool_save_interval` | 快照保存间隔（全局步） | 50,000 |
| `pool_pfsp_temperature` | PFSP softmax 温度 | 0.5 |
| `pool_epsilon` | ε-greedy 均匀采样概率 | 0.1 |
| `pool_reward_ema` | 奖励 EMA 衰减系数 | 0.9 |

### Phase C 相关

| 参数 | 含义 | 默认值 |
|---|---|---|
| `pool_eval_groups` | 对照组数量 | 4 |
| `pool_eval_episodes_per_group` | 每组评估 episode 数 | 20 |
| `eval_ippo_agent0_path` | direct IPPO agent0 路径 | None（需手动填） |
| `eval_pool_agent0_path` | pool agent0 路径 | None（需手动填） |

### 典型命令行

```bash
# 方式一：一键全流程
python custom_ippo_pool.py --run-mode full \
  --ppo-model-paths ppo_agent0.pt ppo_agent1.pt ppo_agent2.pt ppo_agent3.pt

# 方式二：分步运行
# Step 1: Bootstrap
python custom_ippo_pool.py --run-mode ippo_bootstrap \
  --ppo-model-paths ppo_agent0.pt ppo_agent1.pt ppo_agent2.pt ppo_agent3.pt

# Step 2: Pool Cycle
python custom_ippo_pool.py --run-mode pool_cycle \
  --pool-initial-checkpoint-dir saved-models/ippo_bootstrap

# Step 3: Evaluation
python custom_ippo_pool.py --run-mode evaluate \
  --eval-ippo-agent0-path saved-models/ippo_direct_agent0.pt \
  --eval-pool-agent0-path saved-models/clean_rl_ippo_agent0.pt \
  --eval-opponent-checkpoint-dir saved-models/ippo_bootstrap
```
