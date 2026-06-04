# PPT 章节规划：基于机器学习的对抗性机器博弈研究

## Page 1: 封面
- **Page Type**: Cover
- **Page Title**: 基于机器学习的对抗性机器博弈研究
- **Page Subtitle**: 本科毕业设计答辩
- **Selected Template**: cover/tech/039.tpl

- **Content Structure**:
  - **主标题**：基于机器学习的对抗性机器博弈研究
  - **副标题**：华北电力大学本科毕业设计答辩
  - **信息**：答辩人：蒋伦吉 | 专业：计算机科学与技术 | 日期：2026年6月
  - **装饰**：简洁科技风格，主标题居中，底部信息栏

- **Content Density**: Light
- **Narrative Role**: 建立专业第一印象，明确主题与答辩身份

- **Image Requirements**: 无

- **Page Weight**: Core page

---

## Page 2: 目录
- **Page Type**: TOC
- **Page Title**: 答辩内容概览
- **Page Subtitle**: 研究框架与章节安排
- **Selected Template**: toc/tech/3517.tpl

- **Content Structure**:
  - **章节1**：研究背景与意义 — 为什么研究多智能体博弈
  - **章节2**：相关工作 — MARL 与博弈动力学研究现状
  - **章节3**：环境设计 — Tiny Swords 2D 俯视角竞技场
  - **章节4**：方法与算法 — PBRS / 课程学习 / GRU-MLP
  - **章节5**：实验设计与结果 — 三种训练范式对比
  - **章节6**：结论与展望 — 主要贡献与后续方向

- **Content Density**: Medium
- **Narrative Role**: 提供导航地图，让听众预期整体结构

- **Image Requirements**: 无

- **Page Weight**: Transition page

---

## Page 3: 研究背景与意义
- **Page Type**: Transition
- **Page Title**: 研究背景与意义
- **Page Subtitle**: 多智能体博弈为何重要
- **Selected Template**: transition/tech/517.tpl

- **Content Structure**:
  - **RL 成就里程碑**：AlphaGo（2016）击败李世石 → OpenAI Five（2019）Dota 2 世界冠军 → 证明深度学习在复杂博弈中的突破性表现
  - **博弈是试金石**：博弈天然要求智能体具备战略思维、适应性学习、泛化能力，是检验 AI 水平的终极平台
  - **多智能体 = 博弈动力学问题**：多智能体强化学习天然涉及非零和博弈、非平稳环境、信用分配难题
  - **现有困境**：现有 MARL 算法在自由对抗场景中策略易陷入循环、退化或过拟合
  - **本文核心命题**：通过构建可控博弈环境 + 系统化训练范式对比，揭示多智能体对抗训练的规律
  - **三层面贡献**：环境（实验平台）、方法（两阶段训练范式）、理论（策略涌现条件）

- **Content Density**: Medium
- **Narrative Role**: 建立问题意识，说明研究的必要性和价值

- **Image Requirements**: 无

- **Page Weight**: Core page

---

## Page 4: 相关工作与理论基础
- **Page Type**: Content
- **Page Title**: 相关工作与理论基础
- **Page Subtitle**: MARL 与博弈动力学研究现状
- **Selected Template**: content/tech/1581.tpl

- **Content Structure**:
  - **多智能体强化学习分类**：
    - CTDE（集中式训练分布式执行）：如 QMIX、VDN，训练时共享信息，执行时独立
    - DTDE（分布式训练分布式执行）：如 IPPO，完全独立训练，无集中式 critic
  - **博弈论视角**：非零和博弈、纳什均衡、Fictitious Play（虚拟博弈）
  - **平均策略博弈**：训练时固定对手为所有历史策略的平均，提供稳定训练信号
  - **对手池博弈（PFSP）**：动态采样历史策略作为对手，期望覆盖策略空间
  - **奖励塑形（PBRS）**：基于势能函数的 shaping 奖励，保持策略最优性不变
  - **课程学习**：从简单任务渐进到复杂任务，降低探索难度

- **Content Density**: Medium
- **Narrative Role**: 建立理论框架，说明本文方法在学术脉络中的位置

- **Image Requirements**: 无

- **Page Weight**: Secondary page

---

## Page 5: 环境设计 — 整体架构
- **Page Type**: Content
- **Page Title**: 环境设计 — 整体架构
- **Page Subtitle**: Tiny Swords 2D 俯视角竞技场
- **Selected Template**: content/tech/1671.tpl

- **Content Structure**:
  - **游戏类型**：2D 俯视角四人混战竞技场，支持 2-4 智能体同屏对抗
  - **引擎**：Godot 4.x + Godot RL Agents 集成框架
  - **地图设计**：封闭竞技场，包含障碍物、奖励球刷新点、野怪出生点
  - **核心机制**：
    - 攻击系统：扇形判定区域，有冷却时间
    - 饥饿机制：不进食随时间扣血，鼓励主动觅食
    - 奖励球：随机刷新，提供即时奖励信号
    - 野怪系统：中立单位，击杀提供额外奖励
  - **设计目标**：创造具有博弈性的动态环境，支持策略循环涌现

- **Content Density**: Medium
- **Narrative Role**: 展示环境设计能力，说明实验平台的构建

- **Image Requirements**: 需要环境架构图或游戏截图，展示整体布局

- **Page Weight**: Core page

- **Content Page Selection Rationale**: 环境架构是论文三大贡献之首，视觉化展示有助于理解实验平台

---

## Page 6: 智能体设计与观测空间
- **Page Type**: Content
- **Page Title**: 智能体设计与观测空间
- **Page Subtitle**: 四角色差异化设计
- **Selected Template**: content/tech/1687.tpl

- **Content Structure**:
  - **角色差异化**（石头剪刀布式非传递博弈）：
    - **战士**：均衡型，攻击力 15 / 移速 120 / 血量 100 / 射程 60
    - **刺客**：高速高攻脆皮，攻击力 20 / 移速 180 / 血量 60 / 射程 50
    - **骑士**：高坦度长臂，攻击力 12 / 移速 80 / 血量 150 / 射程 80
    - **游侠**：最远射程，攻击力 10 / 移速 100 / 血量 80 / 射程 100
  - **动作空间**：6 个离散动作（上/下/左/右/无操作/攻击），朝向由最近移动方向决定
  - **观测空间（174维分段式）**：
    - SELF：自身 x/y、血量、攻击力、冷却、朝向、饥饿时间（6维）
    - PLAYER：其他玩家的 x/y、血量、攻击力、状态（每玩家 5维 × 3）
    - BALL：视野内奖励球 x/y（每球 2维，最多 4 球）
    - ENEMY：视野内野怪 x/y、血量、攻击力、状态（每怪物 5维，最多 2 个）
    - MAP：自身与边界距离、射线检测障碍物距离（6维）

- **Content Density**: Heavy
- **Narrative Role**: 展示智能体设计的细致程度，强调角色差异化策略

- **Image Requirements**: 角色属性对比表或雷达图

- **Page Weight**: Core page

- **Content Page Selection Rationale**: 智能体设计细节是理解实验结果的基础，数据表格能清晰展示角色差异

---

## Page 7: 奖励塑形与课程学习
- **Page Type**: Content
- **Page Title**: 奖励塑形与课程学习
- **Page Subtitle**: 解决稀疏奖励与探索难题
- **Selected Template**: content/tech/1683.tpl

- **Content Structure**:
  - **稀疏奖励问题**：原始奖励信号稀疏（击杀才得分），智能体难以学习
  - **PBRS 奖励塑形**：
    - 基于势能函数 Φ(s) 定义 shaping reward：F(s,a,s') = γΦ(s') - Φ(s)
    - 保证策略最优性不变（potential-based shaping）
    - 具体设计：距离奖励、朝向奖励、血量变化奖励、击杀奖励
  - **课程学习三阶段**：
    - S1：仅吃球，学习基础移动与觅食
    - S2：引入饥饿机制，学习资源管理
    - S3：加入野怪，学习战斗
    - S4：完整战斗（玩家对抗），学习策略博弈
  - **效果**：课程学习显著提升训练稳定性，减少早期失败率

- **Content Density**: Medium
- **Narrative Role**: 展示解决核心工程难题的方法论

- **Image Requirements**: 课程学习渐进图（S1→S2→S3→S4）

- **Page Weight**: Secondary page

- **Content Page Selection Rationale**: 课程学习曲线是论文方法部分的亮点，可视化展示增强说服力

---

## Page 8: 网络架构设计
- **Page Type**: Content
- **Page Title**: 网络架构设计
- **Page Subtitle**: GRU-MLP 时序网络
- **Selected Template**: content/tech/1583.tpl

- **Content Structure**:
  - **问题**：部分可观测 POMDP，智能体无法获取全局信息
  - **解决方案**：GRU-MLP 时序记忆网络
  - **架构细节**：
    - SELF/PLAYER/ENEMY/MAP 四段时序特征 → concat → GRU（L=2, H=128）
    - BALL 段独立 MLP 处理（非时序信息）
    - GRU 前可选 LayerNorm 稳定训练
    - 输出层：Actor（6 类动作 logits）+ Critic（标量值函数）
  - **设计原则**：固定槽位向量观测直入 GRU，不加前置 MLP 投影，保持时序位置一致性
  - **对比基线**：MLP（无记忆）、Segmented MLP（分段 MLP，无时序建模）

- **Content Density**: Medium
- **Narrative Role**: 展示网络架构设计决策及其工程考量

- **Image Requirements**: 网络架构图（GRU-MLP 结构）

- **Page Weight**: Secondary page

- **Content Page Selection Rationale**: 网络架构是论文技术贡献之一，架构图帮助理解时序建模方式

---

## Page 9: 实验设置与评估方法
- **Page Type**: Content
- **Page Title**: 实验设置与评估方法
- **Page Subtitle**: 统一基准与统计检验
- **Selected Template**: content/tech/1681.tpl

- **Content Structure**:
  - **统一训练步数**：800 万步（三种方法均以此基准评估）
  - **评估对手**：20 组对手池策略，覆盖不同训练阶段
  - **统计方法**：
    - Bootstrap 95% 置信区间（10,000 次重采样）
    - 配对 t 检验（比较方法间差异显著性）
    - Cohen's d 效应量（评估实际差异大小）
  - **核心指标**：每 episode 累计奖励（均值 + 标准差）
  - **排名机制**：每 episode 按 4 智能体 reward 排名，判断是否获胜

- **Content Density**: Light
- **Narrative Role**: 说明实验的科学性和可复现性

- **Image Requirements**: 无

- **Page Weight**: Secondary page

---

## Page 10: 单智能体实验结果
- **Page Type**: Content
- **Page Title**: 单智能体实验结果
- **Page Subtitle**: Phase 1 课程学习验证
- **Selected Template**: content/tech/1680.tpl

- **Content Structure**:
  - **Phase 1 目标**：验证课程学习有效性，为后续多智能体训练奠定基础
  - **S1→S4 学习曲线**：
    - S1（仅吃球）：快速收敛，平均奖励 200+
    - S2（饥饿机制）：轻微波动后稳定
    - S3（加入野怪）：需要更长时间适应
    - S4（完整战斗）：策略复杂度最高，收敛最慢
  - **网络对比**：
    - GRU-MLP 在 S4 阶段表现最优，验证记忆机制价值
    - MLP 在简单阶段（S1-S2）表现尚可，S3-S4 落后
    - Segmented MLP 介于两者之间
  - **关键发现**：课程学习显著降低探索难度，GRU 时序建模对复杂阶段至关重要

- **Content Density**: Medium
- **Narrative Role**: 展示单智能体阶段成果，为多智能体实验做铺垫

- **Image Requirements**: 学习曲线图（S1-S4 奖励曲线 + GRU/MLP/SegMLP 对比）

- **Page Weight**: Core page

- **Content Page Selection Rationale**: 单智能体结果是多智能体实验的基础，曲线图直观展示课程学习效果

---

## Page 11: 多智能体实验结果 — 方法对比
- **Page Type**: Content
- **Page Title**: 多智能体实验结果 — 方法对比
- **Page Subtitle**: 三种训练范式系统对比
- **Selected Template**: content/tech/1581.tpl

- **Content Structure**:
  - **均奖励排名**（800 万步评估）：
    - Average（平均策略）：583.9 ± 12.3
    - IPPO（独立 PPO）：538.2 ± 15.7
    - Pool（PFSP 对手池）：339.0 ± 28.4
  - **统计显著性**：
    - Average vs IPPO：p < 0.01，Cohen's d = 0.45（中等效应）
    - Average vs Pool：p < 0.01，Cohen's d = 1.12（大效应）
    - IPPO vs Pool：p < 0.01，Cohen's d = 0.89（大效应）
  - **获胜率分解**：
    - Average 在 14/20 组对手中胜率 > 50%
    - IPPO 在 12/20 组对手中胜率 > 50%
    - Pool 仅 6/20 组对手中胜率 > 50%

- **Content Density**: Medium
- **Narrative Role**: 呈现核心实验结果，展示方法间显著差异

- **Image Requirements**: 柱状图/箱线图对比三种方法奖励分布

- **Page Weight**: Core page

- **Content Page Selection Rationale**: 这是论文的核心结论页面，数据可视化是最有效的呈现方式

---

## Page 12: 关键发现与洞察
- **Page Type**: Content
- **Page Title**: 关键发现与洞察
- **Page Subtitle**: 策略涌现的动力学特征
- **Selected Template**: content/tech/1690.tpl

- **Content Structure**:
  - **策略循环涌现**：观察到的策略 Rock-Paper-Scissors 循环，没有单一最优策略
  - **非传递性竞争**：战士克制骑士、刺客克制战士、游侠克制刺客的循环关系
  - **平均策略的稳定性**：通过平滑历史策略分布，提供更稳定的训练目标
  - **PFSP 的不稳定性**：
    - 训练曲线高波动性（标准差 28.4 远高于其他方法）
    - 策略循环导致采样分布震荡
    - ε-greedy（10% 均匀采样）不足以保证探索充分性
  - **IPPO 的局限性**：完全分布式训练导致策略非平稳性，收敛到局部均衡

- **Content Density**: Medium
- **Narrative Role**: 从数据上升到理论洞察，提炼核心发现

- **Image Requirements**: 策略循环示意图或 Radar 图

- **Page Weight**: Core page

- **Content Page Selection Rationale**: 理论洞察是论文区别于纯实验报告的关键，需要视觉化辅助理解

---

## Page 13: 局限性与后续工作
- **Page Type**: Content
- **Page Title**: 局限性与后续工作
- **Page Subtitle**: 未完成的工作与改进方向
- **Selected Template**: content/tech/1689.tpl

- **Content Structure**:
  - **联盟训练机制**：因工程复杂度未实现，未来可探索通信机制或团队奖励
  - **PFSP 不稳定性**：采样策略（温度参数、ε-greedy 比例）有待调优
  - **评估深度不足**：仅 20 组对手，可扩展到更多对手验证泛化性
  - **环境规模限制**：4 智能体混战，可扩展到 6-8 智能体测试扩展性
  - **后续方向**：
    - 用户计划转向单智能体 RL 项目（马里奥/贪吃蛇）
    - 将工程经验（环境设计、checkpoint 管理、规则预设）复用到新项目
  - **经验教训**：step() info 设计、tensorboard 保存、博弈规则预设需提前规划

- **Content Density**: Light
- **Narrative Role**: 诚实展示研究局限，体现学术严谨性

- **Image Requirements**: 无

- **Page Weight**: Secondary page

---

## Page 14: 结论与致谢
- **Page Type**: Ending
- **Page Title**: 结论与致谢
- **Page Subtitle**: 感谢聆听，欢迎提问
- **Selected Template**: ending/tech/1017.tpl

- **Content Structure**:
  - **核心贡献总结**：
    - 构建 Tiny Swords 可控博弈环境，支持课程学习与多范式训练
    - 系统对比 IPPO / PFSP / 平均策略博弈三种方法
    - 揭示策略循环、非传递性竞争等博弈动力学特征
  - **关键结论**：平均策略博弈在稳定性和最终性能上表现最优，PFSP 的高波动性揭示了策略循环对训练稳定性的影响
  - **致谢**：感谢指导老师、答辩委员会、同学的支持与帮助
  - **Q&A**：欢迎提问与讨论

- **Content Density**: Light
- **Narrative Role**: 总结收尾，留出问答时间

- **Image Requirements**: 无

- **Page Weight**: Ending page

---
