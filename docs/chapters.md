# PPT 章节规划：基于机器学习的对抗性机器博弈研究

## Page 1: 封面
- **Page Type**: Cover
- **Page Title**: 基于机器学习的对抗性机器博弈研究
- **Page Subtitle**: 本科毕业设计答辩
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
- **Page Title**: 内容概览
- **Page Subtitle**: 研究框架与章节安排
- **Content Structure**:
  - **章节1**：意义与目标 — 为什么以及预期达成什么
  - **章节2**：环境设计 — Tiny Swords 2D 俯视角竞技场
  - **章节3**：方法与算法 — PBRS / 课程学习 / GRU-MLP
  - **章节4**：实验设计与结果 — 三种训练范式对比
  - **章节5**：结论与展望 — 主要贡献与后续方向
- **Content Density**: Medium
- **Narrative Role**: 提供导航地图，让听众预期整体结构
- **Image Requirements**: 无
- **Page Weight**: Transition page

---

## Page 3: 意义与目标

- **Page Type**: Transition
- **Page Title**: 意义与目标
- **Content Structure**:
  - 博弈对抗是人工智能的试金石，要求智能体策略具有鲁棒性和泛化性，能适应风格不一的对手
  - 目标1：设计并实现博弈环境
  - 目标2：探索如何进行高效的强化学习训练，进行单智能体训练
  - 目标3：将训练好的智能体放入博弈场景，测试对比几种博弈训练方法（论文框架图）

- **Content Density**: Medium
- **Narrative Role**: 阐述论文主线
- **Image Requirements**:论文框架图
- **Page Weight**: Core page

---

## Page 4: 环境设计 — 整体架构

- **Page Type**: Content
- **Page Title**: 环境设计 
- **Page Subtitle**: Tiny Swords 2D 俯视角竞技场
- **Content Structure**:
  - **游戏类型**：2D 俯视角四人混战竞技场，支持 4 智能体同屏对抗
  - **引擎**：Godot 4.x + Godot RL Agents 集成框架
  - **地图设计**：封闭竞技场，包含障碍物、奖励球刷新点、野怪出生点（竞技场全景图）
  - **核心机制**：
    - 奖励球：蓝球和绿球，拾取即给予奖励（上面全景图的两张局部放大图）
    - 敌人：中立单位，击杀提供额外奖励（上面全景图的局部放大图）
- **Content Density**: Medium
- **Narrative Role**: 展示环境设计能力，说明实验平台的构建
- **Image Requirements**: 游戏截图，展示整体布局
- **Page Weight**: Core page
- **Content Page Selection Rationale**: 环境架构是论文三大贡献之首，视觉化展示有助于理解实验平台

---

## Page 5: 动作，观测和奖励

- **Page Type**: Content
- **Page Title**: 动作，观测和奖励
- **Content Structure**:
  - **动作空间**：6 个离散动作（上/下/左/右/无操作/攻击），朝向由最近移动方向决定（智能体随机探索时的GIF动图）
  - **观测空间（174维分段式）**：
    - SELF：自身位置、血量、攻击力、冷却、朝向（15维）
    - PLAYER：其他玩家的位置、血量、攻击力、状态（每玩家 11维 × 3）
    - BALL：视野内奖励球位置（每球 5维，最多 8 球）
    - ENEMY：视野内敌人位置、血量、攻击力、状态（每怪物 10维，最多 5 个）
    - MAP：自身与边界距离、射线检测障碍物距离（36维）
  - 奖励：
    - 奖励球：基于势能的塑形奖励
    - 稠密奖励：对敌人或玩家造成伤害给予中间奖励，移动，攻击，以及在中央区域活动都给予小额奖励或惩罚
    - 其余：稀疏奖励
    - 不同智能体奖励配置不同（展示表格）
- **Content Density**: medium
- **Image Requirements**:智能体随机探索时的GIF动图，奖励配置表
- **Page Weight**: Core page
- **Content Page Selection Rationale**: 智能体设计细节是理解实验结果的基础，数据表格能清晰展示角色差异

---

## Page 6: 训练方案对比

- **Page Type**: Content
- **Page Title**: 训练方案对比
- **Page Subtitle**: 加速单智能体阶段训练
- **Content Structure**:
  - 有无有效位掩码：（对比图）
  - **PBRS 奖励塑形**：对比不同势函数效果（对比图）
  - 最近球势能与所有球势能计算方式：（对比图）
  - 避障方案对比：（对比图）
  
- **Content Density**: 尽量light，可适当改为多页，不是重点，快速浏览。
- **Narrative Role**: 展示解决核心工程难题的方法论
- **Image Requirements**: 多张对比图
- **Page Weight**: Secondary page
- **Content Page Selection Rationale**: 对比曲线是论文方法部分的亮点，可视化展示增强说服力

---

## Page 7: 网络架构设计

- **Page Type**: Content
- **Page Title**: 网络架构设计
- **Content Structure**:
  - MLP：观测向量直接送入多层感知机，输出层：Actor（6 类动作 logits）+ Critic（标量值函数）
  - 分段 MLP：将不同语义观测段送入不同子MLP，再进行融合，输出层：Actor（6 类动作 logits）+ Critic（标量值函数）
  - GRU-MLP：SELF/PLAYER/ENEMY/MAP 四段时序特征 → concat → GRU，BALL 段独立 MLP 处理，输出层：Actor（6 类动作 logits）+ Critic（标量值函数）（架构图）
  
- **Content Density**: light
- **Narrative Role**: 展示网络架构设计决策及其工程考量
- **Image Requirements**: 网络架构图
- **Page Weight**: Secondary page
- **Content Page Selection Rationale**: 网络架构是论文技术贡献之一，架构图帮助理解时序建模方式

---

## Page 8: 网络架构对比

- **Page Type**: Content
- **Page Title**: 网络架构对比
- **Content Structure**:
  - 奖励球收集任务对比：
  - PvE战斗任务对比：
  - 给出在完整环境中进行参数搜索结果
  - 给出GRU网络性能不佳的猜想
- **Content Density**: Light
- **Narrative Role**: 
- **Image Requirements**: 两张对比图，参数搜索结果分布图，和游戏截图
- **Page Weight**: Secondary page

---

## Page 9:多智能体训练方案对比

- **Page Type**: Content
- **Page Title**: 多智能体训练方案对比
- **Content Structure**:
  - 直接求解博弈均衡难度是ppad完全（邓小铁论文图片），不直接求解。多人非对称混战的一般和博弈场景没有先例研究，将之前用于对称的研究方法用于该环境测试性能。
  - 测试流程：（流程图）
  - 方法讲述：
    - 对手池初始化方法
    - IPPO：经过验证的虽然有理论缺点但是效果好的多智能体方法
    - 对手池博弈：参考AlphaStar基于PSRO框架的联盟训练，但由于是非对称场景去掉了自博弈，由于算力和时间预算去掉了对手池的增长和更新
    - 平均策略博弈：基于虚拟博弈的思想，将20个对手抽象为一个均匀使用多种策略的一个对手，将对手池中所有的对手的动作概率分布进行平均作为该对手的动作。
- **Content Density**:light
- **Narrative Role**: 
- **Image Requirements**: 流程图等
- **Page Weight**: Core page
- **Content Page Selection Rationale**: 

---

## Page 10: 多智能体实验结果

- **Page Type**: Content
- **Page Title**: 多智能体实验结果 — 方法对比
- **Content Structure**:
  - 结果：统计检验结果显示三种方法都导致了显著的策略退化
  - 查看行为分解图，发现每种策略都不攻击其他智能体，三种博弈训练方法攻击敌人次数减少
  - 提出退化猜想，PvE是强基线策略，将其他智能体纳入决策过程会导致策略退化，很难找回PvE主导策略或者挖掘更强策略。
- **Content Density**: light
- **Narrative Role**: 呈现核心实验结果，展示方法间显著差异
- **Image Requirements**: 柱状图/箱线图对比三种方法奖励分布
- **Page Weight**: Core page
- **Content Page Selection Rationale**: 这是论文的核心结论页面，数据可视化是最有效的呈现方式

---

## Page 11:迁移实验

- **Page Type**: Content
- **Page Title**: 迁移实验
- **Content Structure**:
  - 对智能体进行PvP专项训练
  - 把智能体放到完整环境，同时跑两组，一组正常策略更新，一组冻结策略
  - 结果显示PvE在替代PvP
  - 阐述是游戏本身设计不当，主动博弈收益低。

---

## Page 12:感谢页

