const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "蒋伦吉";
pres.title = "基于机器学习的对抗性机器博弈研究 — 答辩幻灯片";

// ============================================================
// Design constants
// ============================================================
const C = {
  bg: "FFFFFF",
  text: "333333",
  textLight: "666666",
  accent: "2B579A",
  accentLight: "D6E4F0",
  line: "CCCCCC",
  lineDark: "999999",
  white: "FFFFFF",
  warn: "C00000",
  good: "2B7A2B",
};

const FONT = "Microsoft YaHei";
const FONT_EN = "Arial";

// Helper: add a thin horizontal line
function addLine(slide, y, color, width) {
  slide.addShape(pres.shapes.LINE, {
    x: 0.6, y: y, w: 8.8, h: 0,
    line: { color: color || C.line, width: width || 0.5 },
  });
}

// Helper: section title bar
function addSectionTitle(slide, text) {
  slide.addText(text, {
    x: 0.6, y: 0.25, w: 8.8, h: 0.55,
    fontSize: 22, fontFace: FONT, color: C.accent,
    bold: true, margin: 0,
  });
  addLine(slide, 0.85, C.accent, 1.5);
}

// Helper: slide number
function addSlideNum(slide, num, total) {
  slide.addText(`${num} / ${total}`, {
    x: 8.5, y: 5.15, w: 1, h: 0.35,
    fontSize: 9, fontFace: FONT_EN, color: C.textLight,
    align: "right", margin: 0,
  });
}

const TOTAL = 18;

// ============================================================
// Slide 1: Title
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Top accent line
  s.addShape(pres.shapes.LINE, {
    x: 0.6, y: 0.5, w: 8.8, h: 0,
    line: { color: C.accent, width: 3 },
  });

  // University & department
  s.addText("华北电力大学  ·  控制与计算机工程学院", {
    x: 0.6, y: 0.75, w: 8.8, h: 0.4,
    fontSize: 12, fontFace: FONT, color: C.textLight,
    align: "center", margin: 0,
  });

  // Title
  s.addText("基于机器学习的对抗性机器博弈研究", {
    x: 0.8, y: 1.85, w: 8.4, h: 0.9,
    fontSize: 32, fontFace: FONT, color: C.text, bold: true,
    align: "center", margin: 0,
  });

  // Subtitle
  s.addText("多智能体博弈环境构建与对抗训练方法对比", {
    x: 0.8, y: 2.7, w: 8.4, h: 0.5,
    fontSize: 16, fontFace: FONT, color: C.accent,
    align: "center", margin: 0,
  });

  // Bottom accent line
  s.addShape(pres.shapes.LINE, {
    x: 3.5, y: 3.4, w: 3, h: 0,
    line: { color: C.lineDark, width: 0.8 },
  });

  // Info
  s.addText([
    { text: "答辩人：蒋伦吉", options: { breakLine: true } },
    { text: "指导教师：刘春阳  副教授", options: { breakLine: true } },
    { text: "专业：计算机科学与技术", options: {} },
  ], {
    x: 0.8, y: 3.65, w: 8.4, h: 1.0,
    fontSize: 14, fontFace: FONT, color: C.textLight,
    align: "center", margin: 0,
  });

  // Date
  s.addText("2026 年 6 月", {
    x: 0.8, y: 4.85, w: 8.4, h: 0.35,
    fontSize: 12, fontFace: FONT_EN, color: C.lineDark,
    align: "center", margin: 0,
  });

  addSlideNum(s, 1, TOTAL);
}

// ============================================================
// Slide 2: Outline
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "目  录");

  const items = [
    ["01", "研究背景与意义", "对抗性博弈场景下的多智能体强化学习"],
    ["02", "环境设计 — Tiny Swords", "174维观测 · 6动作 · PBRS多层奖励 · 课程学习"],
    ["03", "单智能体基础实验", "观测消融 · 奖励塑形 · 避障策略 · 网络架构对比"],
    ["04", "多智能体博弈训练", "IPPO · 对手池博弈 · 平均策略博弈"],
    ["05", "评估结果与统计分析", "三类方法的多维量化对比"],
    ["06", "总结与展望", "主要贡献与未来方向"],
  ];

  items.forEach((item, i) => {
    const y = 1.1 + i * 0.68;
    // Number
    s.addText(item[0], {
      x: 0.6, y: y, w: 0.7, h: 0.55,
      fontSize: 22, fontFace: FONT_EN, color: C.accent,
      bold: true, align: "right", margin: 0,
    });
    // Title
    s.addText(item[1], {
      x: 1.5, y: y, w: 5, h: 0.3,
      fontSize: 16, fontFace: FONT, color: C.text, bold: true, margin: 0,
    });
    // Description
    s.addText(item[2], {
      x: 1.5, y: y + 0.28, w: 7.5, h: 0.25,
      fontSize: 11, fontFace: FONT, color: C.textLight, margin: 0,
    });
    // Divider
    if (i < items.length - 1) {
      addLine(s, y + 0.62, C.line, 0.3);
    }
  });

  // Bottom line
  s.addShape(pres.shapes.LINE, {
    x: 0.6, y: 5.1, w: 8.8, h: 0,
    line: { color: C.accent, width: 1.5 },
  });

  addSlideNum(s, 2, TOTAL);
}

// ============================================================
// Slide 3: Research Background
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "研究背景与意义");

  // Left column - Context
  s.addText("AI 突破往往发生在博弈对抗场景中", {
    x: 0.6, y: 1.1, w: 4.2, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const milestones = [
    { text: "AlphaGo (2016)", options: { bold: true, breakLine: false } },
    { text: " — 围棋超越人类顶尖棋手", options: { breakLine: true } },
    { text: "AlphaStar (2019)", options: { bold: true, breakLine: false } },
    { text: " — 星际争霸II 宗师级", options: { breakLine: true } },
    { text: "OpenAI Five (2019)", options: { bold: true, breakLine: false } },
    { text: " — Dota 2 击败世界冠军", options: {} },
  ];

  s.addText(milestones, {
    x: 0.6, y: 1.55, w: 4.2, h: 1.6,
    fontSize: 12, fontFace: FONT, color: C.text,
    paraSpaceAfter: 6, margin: 0,
  });

  // Right column - Core problem
  s.addText("多智能体博弈的核心挑战", {
    x: 5.2, y: 1.1, w: 4.2, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.55, w: 4.2, h: 2.0,
    fill: { color: C.accentLight },
    line: { color: C.accent, width: 1 },
  });

  const challenges = [
    { text: "非平稳性 (Non-stationarity)", options: { bold: true, breakLine: true } },
    { text: "每个智能体策略更新 → 其他智能体面对的环境转移概率改变 → MDP 假设不再成立", options: { breakLine: true, fontSize: 11 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "CTDE 局限", options: { bold: true, breakLine: true } },
    { text: "集中式训练需全局信息共享 → 对完全对抗场景不可行 → 采用 DTDE (IPPO)", options: { fontSize: 11 } },
  ];

  s.addText(challenges, {
    x: 5.4, y: 1.65, w: 3.8, h: 1.8,
    fontSize: 12, fontFace: FONT, color: C.text,
    paraSpaceAfter: 4, margin: 0,
  });

  // Bottom - Research approach
  s.addText("研究路径", {
    x: 0.6, y: 3.8, w: 8.8, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  addLine(s, 4.15, C.accent, 1);

  const steps = ["环境构建\nGodot + 174维观测", "基础训练\n课程学习+ PBRS", "单智能体\n实验对比", "博弈训练\nIPPO/Pool/Avg", "多维评估\n统计检验"];
  const stepW = 1.6;
  const startX = 0.8;
  steps.forEach((step, i) => {
    const x = startX + i * (stepW + 0.15);
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 4.25, w: stepW, h: 0.7,
      fill: { color: i === 4 ? C.accent : C.white },
      line: { color: C.accent, width: 1 },
    });
    s.addText(step, {
      x: x, y: 4.25, w: stepW, h: 0.7,
      fontSize: 9, fontFace: FONT, color: i === 4 ? C.white : C.accent,
      align: "center", valign: "middle", margin: 0,
    });
    // Arrow
    if (i < steps.length - 1) {
      s.addText("→", {
        x: x + stepW, y: 4.25, w: 0.2, h: 0.7,
        fontSize: 11, fontFace: FONT_EN, color: C.lineDark,
        align: "center", valign: "middle", margin: 0,
      });
    }
  });

  addSlideNum(s, 3, TOTAL);
}

// ============================================================
// Slide 4: Related Work
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "相关理论与技术基础");

  // Three columns
  const cols = [
    {
      title: "PPO 算法",
      items: [
        "Actor-Critic 架构",
        "GAE 广义优势估计",
        "Clipped surrogate objective",
        "限制新旧策略差异",
        "超参数鲁棒 · 样本高效",
        "→ IPPO: 每智能体独立 PPO",
      ],
    },
    {
      title: "奖励塑形 (PBRS)",
      items: [
        "F = γΦ(s') - Φ(s)",
        "势函数 Φ 注入领域知识",
        "不改变 MDP 最优策略",
        "稠密引导替代稀疏终局奖励",
        "加速策略收敛",
        "→ 多层奖励体系设计",
      ],
    },
    {
      title: "博弈论基础",
      items: [
        "自博弈 (Self-Play)",
        "虚拟自我博弈 (FSP/NFSP)",
        "PSRO — 策略空间响应 Oracle",
        "联盟训练 (AlphaStar)",
        "对手池采样 (PFSP)",
        "→ 非平稳性处理框架",
      ],
    },
  ];

  cols.forEach((col, i) => {
    const x = 0.6 + i * 3.1;
    // Title
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.1, w: 2.8, h: 0.4,
      fill: { color: C.accent },
    });
    s.addText(col.title, {
      x: x, y: 1.1, w: 2.8, h: 0.4,
      fontSize: 13, fontFace: FONT, color: C.white,
      align: "center", valign: "middle", bold: true, margin: 0,
    });

    // Items
    const items = col.items.map((item, j) => ({
      text: item,
      options: {
        bullet: { code: "2014" },
        breakLine: j < col.items.length - 1,
        fontSize: item.startsWith("→") ? 11 : 10.5,
        bold: item.startsWith("→"),
        color: item.startsWith("→") ? C.accent : C.text,
      },
    }));

    s.addText(items, {
      x: x + 0.1, y: 1.6, w: 2.65, h: 3.2,
      fontFace: FONT, color: C.text,
      paraSpaceAfter: 5, margin: 0,
    });

    // Right border
    if (i < cols.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: x + 3.0, y: 1.2, w: 0, h: 3.4,
        line: { color: C.line, width: 0.5 },
      });
    }
  });

  addSlideNum(s, 4, TOTAL);
}

// ============================================================
// Slide 5: Tiny Swords Environment
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "博弈环境 — Tiny Swords");

  // Left: Key specs
  const specs = [
    ["游戏类型", "2D 俯视角四人混战竞技场"],
    ["引擎", "Godot 4.x + Godot RL Agents"],
    ["地图", "18×18 栅格正方形竞技场"],
    ["局时长", "3 分钟 / 1350 动作步"],
    ["智能体", "4 个，各出生在四角"],
    ["敌人 NPC", "最多 5 个，随机游走 + 追杀"],
    ["奖励球", "A 类(出生点) + B 类(中央，可重生)"],
    ["视野", "半径 200px，部分可观测"],
  ];

  specs.forEach((spec, i) => {
    const y = 1.1 + i * 0.43;
    s.addText(spec[0], {
      x: 0.6, y: y, w: 1.4, h: 0.32,
      fontSize: 11, fontFace: FONT, color: C.accent, bold: true, margin: 0,
    });
    s.addText(spec[1], {
      x: 2.1, y: y, w: 4.2, h: 0.32,
      fontSize: 11, fontFace: FONT, color: C.text, margin: 0,
    });
    if (i < specs.length - 1) {
      addLine(s, y + 0.38, C.line, 0.2);
    }
  });

  // Right: Agent classes
  s.addText("智能体差异化风格", {
    x: 6.0, y: 1.1, w: 3.6, h: 0.35,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const agents = [
    ["战士型", "Agent 0, 2", "进攻导向", "高战斗收益 · 低采集收益"],
    ["采集型", "Agent 1", "避战收集", "高拾球收益 · 低战斗收益"],
    ["均衡型", "Agent 3", "保守稳健", "中间值 · 低风险得分"],
  ];

  agents.forEach((ag, i) => {
    const y = 1.65 + i * 1.05;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.0, y: y, w: 3.6, h: 0.85,
      fill: { color: C.white },
      line: { color: C.accent, width: 1 },
    });
    s.addText(ag[0], {
      x: 6.15, y: y + 0.05, w: 3.3, h: 0.28,
      fontSize: 12, fontFace: FONT, color: C.accent, bold: true, margin: 0,
    });
    s.addText([
      { text: ag[1] + "  ·  ", options: { fontSize: 10, color: C.textLight } },
      { text: ag[2], options: { fontSize: 10, bold: true } },
    ], {
      x: 6.15, y: y + 0.3, w: 3.3, h: 0.22,
      fontFace: FONT, color: C.text, margin: 0,
    });
    s.addText(ag[3], {
      x: 6.15, y: y + 0.5, w: 3.3, h: 0.22,
      fontSize: 10, fontFace: FONT, color: C.textLight, margin: 0,
    });
  });

  // Bottom: Architecture
  s.addText("训练架构", {
    x: 0.6, y: 4.15, w: 8.8, h: 0.35,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  s.addText("Godot 环境 ← TCP Socket → Python PPO 训练端  ·  CleanRL 风格手写实现  ·  每 8 物理帧一次决策", {
    x: 0.6, y: 4.5, w: 8.8, h: 0.35,
    fontSize: 11, fontFace: FONT, color: C.textLight, align: "center", margin: 0,
  });

  addSlideNum(s, 5, TOTAL);
}

// ============================================================
// Slide 6: Observation & Action Space
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "观测空间与动作空间设计");

  // Observation summary
  s.addText("174 维分段式观测空间", {
    x: 0.6, y: 1.0, w: 5, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const obsRows = [
    [
      { text: "分段", options: { bold: true, color: C.white, fill: { color: C.accent } } },
      { text: "维度", options: { bold: true, color: C.white, fill: { color: C.accent } } },
      { text: "内容", options: { bold: true, color: C.white, fill: { color: C.accent } } },
    ],
    ["SELF", "15", "坐标 · 血量 · 朝向 · 冷却 · 速度 · 上一步动作"],
    ["PLAYER", "33 (3×11)", "对手相对位置 · 血量 · 朝向 · 速度 · 有效掩码"],
    ["BALL", "40 (8×5)", "奖励球相对位置 · 类型 · 距离排序"],
    ["ENEMY", "50 (5×10)", "敌人相对位置 · 血量 · 攻击状态"],
    ["LiDAR", "36 (36×1)", "36 条射线检测障碍物距离"],
  ];

  s.addTable(obsRows, {
    x: 0.6, y: 1.4, w: 4.8,
    fontSize: 9.5, fontFace: FONT,
    border: { pt: 0.5, color: C.line },
    colW: [1.2, 0.8, 2.8],
    rowH: [0.32, 0.28, 0.28, 0.28, 0.28, 0.28],
    margin: [2, 4, 2, 4],
    autoPage: false,
  });

  // Action space
  s.addText("6 种离散动作", {
    x: 6.0, y: 1.0, w: 3.6, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const actions = ["↑ 上移", "↓ 下移", "← 左移", "→ 右移", "⏸ 待机", "⚔ 攻击"];
  actions.forEach((act, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 6.0 + col * 1.8;
    const y = 1.5 + row * 0.55;
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: 1.6, h: 0.42,
      fill: { color: C.white },
      line: { color: C.accent, width: 1 },
    });
    s.addText(act, {
      x: x, y: y, w: 1.6, h: 0.42,
      fontSize: 11, fontFace: FONT, color: C.text,
      align: "center", valign: "middle", margin: 0,
    });
  });

  // Key design principles
  s.addText("编码设计原则", {
    x: 0.6, y: 3.3, w: 8.8, h: 0.35,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const principles = [
    { title: "以自身为中心", desc: "坐标 → 相对偏移量，除以视野归一化 [-1,1]，保证平移不变性" },
    { title: "固定槽位编码", desc: "PLAYER 按 ID 固定槽 · BALL 按距离排序 · 保证时序位置一致性" },
    { title: "有效位掩码", desc: "每个实体槽位末尾标记有效位 0/1，区分有效信息与填充零" },
    { title: "动力学信息", desc: "攻击动画 · 冷却比例 · 速度 · 上一动作 one-hot — 捕获瞬时动态" },
  ];

  principles.forEach((p, i) => {
    const x = 0.6 + (i % 2) * 4.5;
    const y = 3.7 + Math.floor(i / 2) * 0.6;
    s.addText(p.title, {
      x: x, y: y, w: 1.8, h: 0.25,
      fontSize: 10.5, fontFace: FONT, color: C.accent, bold: true, margin: 0,
    });
    s.addText(p.desc, {
      x: x + 1.85, y: y, w: 2.5, h: 0.45,
      fontSize: 9.5, fontFace: FONT, color: C.textLight, margin: 0,
    });
    // Divider
    s.addShape(pres.shapes.LINE, {
      x: x + 1.8, y: y + 0.02, w: 0, h: 0.42,
      line: { color: C.line, width: 0.5 },
    });
  });

  addSlideNum(s, 6, TOTAL);
}

// ============================================================
// Slide 7: Reward Design
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "奖励函数设计 — 基于 PBRS 的多层体系");

  // Left: PBRS theory
  s.addText("PBRS 理论基础", {
    x: 0.6, y: 1.1, w: 4.2, h: 0.35,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.5, w: 4.2, h: 1.1,
    fill: { color: C.accentLight },
    line: { color: C.accent, width: 1 },
  });

  s.addText([
    { text: "F(s, a, s') = γ·Φ(s') − Φ(s)", options: { bold: true, breakLine: true, fontSize: 13 } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "▸ 塑形奖励 = 势函数的折扣差分", options: { breakLine: true, fontSize: 10.5 } },
    { text: "▸ 不改变 MDP 的最优策略 (Ng et al., 1999)", options: { breakLine: true, fontSize: 10.5 } },
    { text: "▸ 将领域知识注入学习过程，稠密引导替代稀疏奖励", options: { fontSize: 10.5 } },
  ], {
    x: 0.8, y: 1.55, w: 3.8, h: 1.0,
    fontFace: FONT, color: C.text, margin: 0,
  });

  // Right: Reward structure
  s.addText("双层奖励体系", {
    x: 5.2, y: 1.1, w: 4.2, h: 0.35,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const rewardItems = [
    { label: "事件稀疏奖励", items: "拾球 · 攻击伤害 · 击杀 · 受伤 · 死亡" },
    { label: "势能稠密塑形", items: "Φ(d) = −d (线性) · 引导走向奖励球" },
    { label: "撞墙惩罚", items: "−0.5/次 (稀疏) · 平衡得分与避障" },
    { label: "机动惩罚", items: "移动 −0.001/帧 · 静止 −0.005/帧 · 防止惰性" },
    { label: "中心区域奖励", items: "+0.003/帧 · 鼓励进入资源密集区" },
    { label: "攻击惩罚", items: "−0.02/次 · 防止无意义攻击" },
  ];

  rewardItems.forEach((item, i) => {
    const y = 1.5 + i * 0.55;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y: y, w: 4.2, h: 0.48,
      fill: { color: C.white },
      line: { color: C.line, width: 0.5 },
    });
    s.addText(item.label, {
      x: 5.35, y: y + 0.02, w: 1.6, h: 0.2,
      fontSize: 10, fontFace: FONT, color: C.accent, bold: true, margin: 0,
    });
    s.addText(item.items, {
      x: 5.35, y: y + 0.22, w: 3.9, h: 0.22,
      fontSize: 9.5, fontFace: FONT, color: C.textLight, margin: 0,
    });
  });

  // Bottom: Normalization
  s.addText("Welford 在线 z-score 归一化到 [−1, 1]  ·  零和约束：战士=采集+均衡", {
    x: 0.6, y: 4.85, w: 8.8, h: 0.35,
    fontSize: 11, fontFace: FONT, color: C.textLight, align: "center", margin: 0,
  });

  addSlideNum(s, 7, TOTAL);
}

// ============================================================
// Slide 8: Curriculum Learning
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "课程学习 — 五阶段渐进式训练");

  const stages = [
    { name: "S1", title: "无障碍拾球", steps: "50万", desc: "无墙、无敌人 — 仅学拾球", color: "A8D8EA" },
    { name: "S2", title: "避障+拾球", steps: "+100万", desc: "引入障碍物 — 学习避障", color: "AA96DA" },
    { name: "S3", title: "敌人战斗", steps: "+200万", desc: "引入敌人 — 学习战斗", color: "FCBAD3" },
    { name: "S4", title: "综合场景", steps: "+100万", desc: "敌人+障碍+球 — 综合训练", color: "FFD3B6" },
    { name: "Main", title: "完整环境", steps: "+50万", desc: "四角出生点 · 最终部署", color: "D5E8D4" },
  ];

  stages.forEach((st, i) => {
    const x = 0.6 + i * 1.82;
    // Stage box
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.1, w: 1.65, h: 2.8,
      fill: { color: st.color, transparency: 40 },
      line: { color: C.lineDark, width: 1 },
    });
    // Stage name
    s.addText(st.name, {
      x: x, y: 1.15, w: 1.65, h: 0.45,
      fontSize: 20, fontFace: FONT_EN, color: C.accent, bold: true,
      align: "center", margin: 0,
    });
    // Title
    s.addText(st.title, {
      x: x + 0.05, y: 1.7, w: 1.55, h: 0.3,
      fontSize: 12, fontFace: FONT, color: C.text, bold: true,
      align: "center", margin: 0,
    });
    // Steps
    s.addText(st.steps, {
      x: x + 0.05, y: 2.1, w: 1.55, h: 0.25,
      fontSize: 10, fontFace: FONT_EN, color: C.accent,
      align: "center", margin: 0,
    });
    // Description
    s.addText(st.desc, {
      x: x + 0.05, y: 2.5, w: 1.55, h: 0.9,
      fontSize: 9.5, fontFace: FONT, color: C.textLight,
      align: "center", margin: 0,
    });

    // Arrow between stages
    if (i < stages.length - 1) {
      s.addText("→", {
        x: x + 1.65, y: 1.1, w: 0.2, h: 2.8,
        fontSize: 16, fontFace: FONT_EN, color: C.lineDark,
        align: "center", valign: "middle", margin: 0,
      });
    }
  });

  // Total steps
  s.addText("累计训练步数：500 万步", {
    x: 0.6, y: 4.1, w: 8.8, h: 0.35,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, align: "center", margin: 0,
  });

  // Training specifics
  s.addText("每阶段继承上一阶段最优检查点  ·  PP0 算法  ·  4 智能体同步训练", {
    x: 0.6, y: 4.6, w: 8.8, h: 0.35,
    fontSize: 11, fontFace: FONT, color: C.textLight, align: "center", margin: 0,
  });

  addSlideNum(s, 8, TOTAL);
}

// ============================================================
// Slide 9: Single-Agent Experiments Overview
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "单智能体实验 — 四大对比维度");

  const exps = [
    {
      title: "观测空间消融",
      question: "有效掩码是否必要？",
      methods: "有掩码 vs 无掩码 (174维 vs 158维)",
      result: "有掩码总得分 +9.9%\nAgent3 改善最显著 +48.3%",
    },
    {
      title: "奖励球塑形方案",
      question: "哪种势函数最优？",
      methods: "线性 / 指数 / 反比例 / 距离奖励 / 稀疏",
      result: "距离奖励 390.1 最高\n线性 PBRS 379.4 次之\n最近球 vs 所有球: +67%",
    },
    {
      title: "障碍物避障策略",
      question: "如何平衡得分与避障？",
      methods: "稀疏惩罚(0.05/0.5/5.0) + 稠密塑形(PBRS/直接)",
      result: "惩罚0.5: 撞墙−29% 得分持平\n稠密塑形撞墙−54% 但得分微降",
    },
    {
      title: "网络架构对比",
      question: "MLP/SegMLP/GRU-MLP?",
      methods: "拾球任务 + 战斗任务 + Optuna搜索",
      result: "战斗: MLP 775.2 最优\nOptuna: SegMLP 1116.3\nGRU 表现持续落后",
    },
  ];

  exps.forEach((exp, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.6 + col * 4.6;
    const y = 1.0 + row * 2.0;

    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: 4.2, h: 1.8,
      fill: { color: C.white },
      line: { color: i === 0 ? C.accent : C.line, width: i === 0 ? 1.5 : 0.8 },
    });

    // Number
    s.addText(`0${i + 1}`, {
      x: x, y: y + 0.05, w: 0.6, h: 0.5,
      fontSize: 22, fontFace: FONT_EN, color: C.accent, bold: true,
      align: "right", margin: 0,
    });

    // Title
    s.addText(exp.title, {
      x: x + 0.7, y: y + 0.1, w: 3.3, h: 0.32,
      fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
    });

    // Separator
    s.addShape(pres.shapes.LINE, {
      x: x + 0.2, y: y + 0.5, w: 3.8, h: 0,
      line: { color: C.line, width: 0.5 },
    });

    // Question
    s.addText(exp.question, {
      x: x + 0.2, y: y + 0.55, w: 3.8, h: 0.25,
      fontSize: 10, fontFace: FONT, color: C.textLight, italic: true, margin: 0,
    });

    // Methods
    s.addText(exp.methods, {
      x: x + 0.2, y: y + 0.8, w: 3.8, h: 0.22,
      fontSize: 9.5, fontFace: FONT, color: C.text, margin: 0,
    });

    // Result
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.15, y: y + 1.1, w: 3.9, h: 0.55,
      fill: { color: C.accentLight },
      line: { color: "000000", width: 0 },  // no border
    });
    s.addText(exp.result, {
      x: x + 0.25, y: y + 1.12, w: 3.7, h: 0.5,
      fontSize: 9.5, fontFace: FONT, color: C.accent,
      margin: 0,
    });
  });

  addSlideNum(s, 9, TOTAL);
}

// ============================================================
// Slide 10: Key Results 1 - Obs Mask & Reward Shaping
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "实验结果 ① — 观测消融与奖励塑形");

  // Chart 1: Observation ablation (bar chart)
  s.addText("有效掩码消融 — 总得分对比", {
    x: 0.6, y: 1.0, w: 4.2, h: 0.3,
    fontSize: 12, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  s.addChart(pres.charts.BAR, [
    { name: "无掩码", labels: ["Agent0", "Agent1", "Agent2", "Agent3", "总平均"],
      values: [75.2, 72.5, 65.9, 41.0, 254.7] },
    { name: "有掩码", labels: ["Agent0", "Agent1", "Agent2", "Agent3", "总平均"],
      values: [77.6, 73.3, 68.2, 60.8, 279.8] },
  ], {
    x: 0.6, y: 1.3, w: 4.2, h: 2.3,
    barDir: "col",
    chartColors: [C.lineDark, C.accent],
    showLegend: true,
    legendPos: "b",
    legendFontSize: 9,
    catAxisLabelColor: C.textLight,
    valAxisLabelColor: C.textLight,
    valGridLine: { color: C.line, size: 0.3 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: C.white } },
    showValue: false,
  });

  // Chart 2: Reward shaping comparison
  s.addText("奖励塑形方案对比 — 总得分", {
    x: 5.2, y: 1.0, w: 4.2, h: 0.3,
    fontSize: 12, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  s.addChart(pres.charts.BAR, [
    { name: "总平均", labels: ["线性", "指数", "反比例", "距离奖励", "稀疏"],
      values: [379.4, 310.1, 380.2, 390.1, 234.0] },
  ], {
    x: 5.2, y: 1.3, w: 4.2, h: 2.3,
    barDir: "col",
    chartColors: [C.accent],
    showLegend: false,
    catAxisLabelColor: C.textLight,
    valAxisLabelColor: C.textLight,
    valGridLine: { color: C.line, size: 0.3 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: C.white } },
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelColor: C.text,
    dataLabelFontSize: 9,
  });

  // Bottom: Key takeaways
  s.addText("关键结论", {
    x: 0.6, y: 3.7, w: 8.8, h: 0.3,
    fontSize: 12, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  s.addText([
    { text: "① ", options: { bold: true, color: C.accent } },
    { text: "有效掩码提升 9.9%，Agent3 改善幅度最大 (+48.3%) — 区分有效/无效信息对弱势出生点尤为关键", options: { breakLine: true } },
    { text: "② ", options: { bold: true, color: C.accent } },
    { text: "线性势函数 (379.4) 与距离奖励 (390.1) 接近，但线性 PBRS 保证最优策略不变性 → 最终选择", options: { breakLine: true } },
    { text: "③ ", options: { bold: true, color: C.accent } },
    { text: "最近球模式高出所有球模式 67% — 将多目标分解为阶段性子目标效率更高", options: {} },
  ], {
    x: 0.6, y: 4.05, w: 8.8, h: 1.1,
    fontSize: 10.5, fontFace: FONT, color: C.text,
    paraSpaceAfter: 8, margin: 0,
  });

  addSlideNum(s, 10, TOTAL);
}

// ============================================================
// Slide 11: Key Results 2 - Obstacle & Architecture
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "实验结果 ② — 避障策略与网络架构");

  // Obstacle strategy
  s.addText("撞墙惩罚力度对比 (100万步训练)", {
    x: 0.6, y: 1.0, w: 4.2, h: 0.3,
    fontSize: 12, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const obsRows = [
    [
      { text: "惩罚值", options: { bold: true, fill: { color: C.accent }, color: C.white } },
      { text: "总得分", options: { bold: true, fill: { color: C.accent }, color: C.white } },
      { text: "撞墙次数", options: { bold: true, fill: { color: C.accent }, color: C.white } },
    ],
    ["0.05", "272.6", "161.5 ↓"],
    ["0.5 ★", "274.1", "114.8 (−29%)"],
    ["5.0", "122.0", "16.7 (过度保守)"],
  ];
  s.addTable(obsRows, {
    x: 0.6, y: 1.4, w: 4.2, fontSize: 10, fontFace: FONT,
    border: { pt: 0.5, color: C.line },
    colW: [1.2, 1.2, 1.8],
    rowH: [0.3, 0.28, 0.28, 0.28],
    margin: [2, 4, 2, 4],
  });

  // Network comparison
  s.addText("网络架构对比", {
    x: 5.2, y: 1.0, w: 4.2, h: 0.3,
    fontSize: 12, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  s.addChart(pres.charts.BAR, [
    { name: "拾球任务(S1)", labels: ["MLP", "SegMLP", "GRU-96", "GRU-128"],
      values: [862.4, 940.6, 894.4, 905.0] },
    { name: "战斗任务(Sarl)", labels: ["MLP", "SegMLP", "GRU-96", "GRU-128"],
      values: [775.2, 714.6, 582.9, 578.5] },
  ], {
    x: 5.2, y: 1.4, w: 4.2, h: 2.2,
    barDir: "col",
    chartColors: ["5B9BD5", "ED7D31"],
    showLegend: true,
    legendPos: "b",
    legendFontSize: 9,
    catAxisLabelColor: C.textLight,
    valAxisLabelColor: C.textLight,
    valGridLine: { color: C.line, size: 0.3 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: C.white } },
    showValue: false,
  });

  // Key takeaways
  s.addText("关键发现", {
    x: 0.6, y: 2.4, w: 8.8, h: 0.3,
    fontSize: 12, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const findings = [
    "MLP 在战斗任务中表现最佳 (775.2) —— 与直觉相反，视野半径大可观测大部分信息，无需历史记忆",
    "Segmented MLP 在拾球任务和 Optuna 搜索中均领先，具有进一步优化潜力",
    "GRU-MLP 表现持续落后，Optuna 仅分配 17 个 trial 后自动放弃 —— 当前环境下时序网络优势有限",
    "惩罚 0.5 实现最佳平衡：撞墙减少 29%，得分不受影响",
  ];
  findings.forEach((f, i) => {
    s.addText([
      { text: `${i + 1}. `, options: { bold: true, color: C.accent } },
      { text: f, options: {} },
    ], {
      x: 0.6, y: 2.75 + i * 0.55, w: 8.8, h: 0.5,
      fontSize: 10.5, fontFace: FONT, color: C.text, margin: 0,
    });
  });

  addSlideNum(s, 11, TOTAL);
}

// ============================================================
// Slide 12: Multi-Agent Methods
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "多智能体博弈 — 三种训练方法对比");

  // Research question
  s.addText("核心问题：已具备基础技能的智能体共处对抗环境时，何种训练机制能产生应对多样化对手的鲁棒策略？", {
    x: 0.6, y: 1.0, w: 8.8, h: 0.5,
    fontSize: 12, fontFace: FONT, color: C.text, italic: true, margin: 0,
  });

  // Three methods
  const methods = [
    {
      name: "IPPO",
      subtitle: "独立 PPO · 基线方法",
      desc: "4 个智能体同时独立运行 PPO，\n其他智能体策略更新 = 环境非平稳\n不引入集中式组件",
      result: "均奖励: 495.0 (±41.8 CI)",
      color: C.lineDark,
    },
    {
      name: "对手池博弈 (Pool)",
      subtitle: "PFSP 优先采样",
      desc: "从对手池中按奖励加权采样对手\n聚焦困难对手 (低奖励=高概率)\nε-greedy 保持探索",
      result: "均奖励: 339.0 (±51.9 CI)",
      color: C.warn,
    },
    {
      name: "平均策略博弈 (Avg)",
      subtitle: "虚拟自我博弈",
      desc: "对手池 N=20 组策略动作概率\n取算术平均构成虚拟对手\n理论: 经典博弈论虚拟博弈",
      result: "均奖励: 586.3 (±44.5 CI)",
      color: C.good,
    },
  ];

  methods.forEach((m, i) => {
    const x = 0.6 + i * 3.1;
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.6, w: 2.8, h: 3.0,
      fill: { color: C.white },
      line: { color: m.color, width: 1.5 },
    });
    // Method name
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.6, w: 2.8, h: 0.45,
      fill: { color: m.color },
    });
    s.addText(m.name, {
      x: x, y: 1.6, w: 2.8, h: 0.45,
      fontSize: 14, fontFace: FONT_EN, color: C.white,
      align: "center", valign: "middle", bold: true, margin: 0,
    });
    // Subtitle
    s.addText(m.subtitle, {
      x: x + 0.1, y: 2.15, w: 2.6, h: 0.25,
      fontSize: 10, fontFace: FONT, color: C.textLight, align: "center", margin: 0,
    });
    // Description
    s.addText(m.desc, {
      x: x + 0.2, y: 2.5, w: 2.4, h: 1.2,
      fontSize: 10, fontFace: FONT, color: C.text, align: "center", margin: 0,
    });
    // Result bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.1, y: 3.95, w: 2.6, h: 0.5,
      fill: { color: m.color, transparency: 85 },
    });
    s.addText(m.result, {
      x: x + 0.1, y: 3.95, w: 2.6, h: 0.5,
      fontSize: 9.5, fontFace: FONT, color: m.color, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });

  // Training step note
  s.addText("每种方法训练 800 万步  ·  统一对手池 (20 组 × 4 智能体) 评估  ·  每组 100 局", {
    x: 0.6, y: 4.85, w: 8.8, h: 0.35,
    fontSize: 11, fontFace: FONT, color: C.textLight, align: "center", margin: 0,
  });

  addSlideNum(s, 12, TOTAL);
}

// ============================================================
// Slide 13: Opponent Pool Construction
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "对手池构建 — 轮换冻结迭代博弈");

  // Process flow
  const steps = [
    { num: "1", title: "同步初始训练", desc: "4 智能体同步 IPPO\n训练 100 万步" },
    { num: "2", title: "冻结对手训练", desc: "冻结 Agent 1/2/3\n仅训练 Agent 0" },
    { num: "3", title: "轮换解冻", desc: "依次解冻下一智能体\n冻结其余 3 个" },
    { num: "4", title: "多轮迭代", desc: "共 8 轮 · 每智能体\n每轮 50 万步" },
    { num: "5", title: "构建对手池", desc: "每智能体选最近\n20 个检查点" },
  ];

  steps.forEach((st, i) => {
    const x = 0.4 + i * 1.92;
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.1, w: 1.7, h: 2.0,
      fill: { color: C.white },
      line: { color: C.accent, width: 1 },
    });
    // Number circle
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.55, y: 1.2, w: 0.55, h: 0.55,
      fill: { color: C.accent },
    });
    s.addText(st.num, {
      x: x + 0.55, y: 1.2, w: 0.55, h: 0.55,
      fontSize: 14, fontFace: FONT_EN, color: C.white, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(st.title, {
      x: x + 0.1, y: 1.85, w: 1.5, h: 0.3,
      fontSize: 11, fontFace: FONT, color: C.accent, bold: true,
      align: "center", margin: 0,
    });
    s.addText(st.desc, {
      x: x + 0.1, y: 2.2, w: 1.5, h: 0.7,
      fontSize: 9.5, fontFace: FONT, color: C.textLight,
      align: "center", margin: 0,
    });
    // Arrow
    if (i < steps.length - 1) {
      s.addText("→", {
        x: x + 1.7, y: 1.1, w: 0.25, h: 2.0,
        fontSize: 16, fontFace: FONT_EN, color: C.lineDark,
        align: "center", valign: "middle", margin: 0,
      });
    }
  });

  // PFSP sampling formula
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 3.4, w: 8.8, h: 1.15,
    fill: { color: C.accentLight },
    line: { color: C.accent, width: 1 },
  });

  s.addText("PFSP 对手采样策略", {
    x: 0.8, y: 3.45, w: 8.4, h: 0.3,
    fontSize: 13, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  s.addText([
    { text: "P(c) ∝ exp(−(R_ema − min R_ema) / T)", options: { bold: true, breakLine: true, fontSize: 11.5 } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "▸ 奖励越低的对手被采样概率越高 → 聚焦困难对手", options: { breakLine: true, fontSize: 10 } },
    { text: "▸ ε = 10% 均匀随机采样 → 保持统计新鲜度", options: { breakLine: true, fontSize: 10 } },
    { text: "▸ EMA 胜率追踪 (0.95 衰减) + ELO K=32", options: { fontSize: 10 } },
  ], {
    x: 0.8, y: 3.75, w: 8.4, h: 0.75,
    fontFace: FONT, color: C.text, margin: 0,
  });

  addSlideNum(s, 13, TOTAL);
}

// ============================================================
// Slide 14: Evaluation Results
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "评估结果 — 多维量化对比 (20 组对手 × 100 局)");

  // Main comparison chart
  s.addChart(pres.charts.BAR, [
    {
      name: "均奖励",
      labels: ["IPPO", "对手池博弈", "平均策略博弈", "初始策略(对照)"],
      values: [495.0, 339.0, 586.3, 603.1],
    },
  ], {
    x: 0.6, y: 1.05, w: 5.0, h: 2.5,
    barDir: "col",
    chartColors: [C.lineDark, C.warn, C.accent, C.textLight],
    showLegend: false,
    catAxisLabelColor: C.textLight,
    catAxisLabelFontSize: 9,
    valAxisLabelColor: C.textLight,
    valGridLine: { color: C.line, size: 0.3 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: C.white } },
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelColor: C.text,
    dataLabelFontSize: 10,
  });

  // Right: Stats table
  const statRows = [
    [
      { text: "方法", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 9 } },
      { text: "均奖励", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 9 } },
      { text: "95% CI", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 9 } },
      { text: "σ", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 9 } },
    ],
    ["IPPO", "495.0", "[474, 516]", "97.3"],
    ["对手池博弈", "339.0", "[314, 366]", "146.1"],
    ["平均策略博弈", "586.3", "[564, 608]", "117.7"],
    ["初始策略对照", "603.1", "[576, 630]", "104.7"],
  ];

  s.addTable(statRows, {
    x: 5.8, y: 1.05, w: 3.8, fontSize: 9, fontFace: FONT,
    border: { pt: 0.5, color: C.line },
    colW: [1.3, 0.8, 1.0, 0.7],
    rowH: [0.32, 0.32, 0.32, 0.32, 0.32],
    margin: [2, 3, 2, 3],
  });

  // Bottom: Key insights
  s.addShape(pres.shapes.LINE, {
    x: 0.6, y: 3.7, w: 8.8, h: 0,
    line: { color: C.accent, width: 1 },
  });

  s.addText("核心发现", {
    x: 0.6, y: 3.8, w: 8.8, h: 0.3,
    fontSize: 13, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const insights = [
    { text: "① ", options: { bold: true, color: C.accent } },
    { text: "三种继续训练方法中，平均策略博弈均奖励最高 (586.3 > 495.0 > 339.0)", options: { breakLine: true } },
    { text: "② ", options: { bold: true, color: C.warn } },
    { text: "对手池博弈标准差最高 (146.1) — PFSP 聚焦采样在多人策略循环中损害泛化稳定性", options: { breakLine: true } },
    { text: "③ ", options: { bold: true, color: C.good } },
    { text: "初始策略均奖励 603.1，与平均策略博弈差异不显著 (p=0.301) — 课程学习已提供强基线", options: {} },
  ];

  s.addText(insights, {
    x: 0.6, y: 4.15, w: 8.8, h: 1.0,
    fontSize: 10.5, fontFace: FONT, color: C.text,
    paraSpaceAfter: 6, margin: 0,
  });

  addSlideNum(s, 14, TOTAL);
}

// ============================================================
// Slide 15: Statistical Analysis
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "统计显著性检验 — 配对 t 检验与效应量");

  // Left: Pairwise comparison table
  const pairRows = [
    [
      { text: "配对比较", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 8.5 } },
      { text: "Δ 奖励", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 8.5 } },
      { text: "p-value", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 8.5 } },
      { text: "Cohen's d", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 8.5 } },
      { text: "显著性", options: { bold: true, fill: { color: C.accent }, color: C.white, fontSize: 8.5 } },
    ],
    ["IPPO vs Pool", "+156.1", "1.39e-8", "+2.10", "***"],
    ["IPPO vs Avg", "−91.3", "1.36e-6", "−1.55", "***"],
    ["IPPO vs Init", "−108.0", "8.48e-7", "−1.60", "***"],
    ["Pool vs Avg", "−247.4", "6.64e-13", "−3.78", "***"],
    ["Pool vs Init", "−264.1", "3.49e-12", "−3.44", "***"],
    ["Avg vs Init", "−16.8", "0.301", "−0.24", "n.s."],
  ];

  s.addTable(pairRows, {
    x: 0.6, y: 1.0, w: 5.8, fontSize: 8.5, fontFace: FONT,
    border: { pt: 0.5, color: C.line },
    colW: [1.5, 0.9, 1.0, 1.1, 0.6],
    rowH: [0.3, 0.28, 0.28, 0.28, 0.28, 0.28, 0.3],
    margin: [2, 3, 2, 3],
  });

  // Right: Effect size visualization
  s.addText("效应量可视化 (Cohen's d)", {
    x: 6.7, y: 1.0, w: 2.7, h: 0.3,
    fontSize: 11, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const effectPairs = [
    { label: "IPPO vs Pool", d: 2.10, color: C.accent },
    { label: "Pool vs Avg", d: -3.78, color: C.warn },
    { label: "IPPO vs Avg", d: -1.55, color: C.accent },
    { label: "Pool vs Init", d: -3.44, color: C.warn },
    { label: "IPPO vs Init", d: -1.60, color: C.accent },
    { label: "Avg vs Init", d: -0.24, color: C.lineDark },
  ];

  effectPairs.forEach((ep, i) => {
    const y = 1.4 + i * 0.58;
    s.addText(ep.label, {
      x: 6.7, y: y, w: 1.3, h: 0.2,
      fontSize: 7.5, fontFace: FONT, color: C.text, margin: 0,
    });
    const barMax = 3.0;
    const barW = Math.min(Math.abs(ep.d) / 4.5 * barMax, barMax);
    const barX = ep.d >= 0 ? 8.0 : 8.0 - barW;
    s.addShape(pres.shapes.RECTANGLE, {
      x: barX, y: y + 0.25, w: barW, h: 0.15,
      fill: { color: ep.color },
    });
    s.addText(ep.d.toFixed(2), {
      x: 9.0, y: y + 0.2, w: 0.5, h: 0.25,
      fontSize: 8, fontFace: FONT_EN, color: ep.color, bold: true, margin: 0,
    });
  });

  // Bottom notes
  s.addShape(pres.shapes.LINE, {
    x: 0.6, y: 4.0, w: 8.8, h: 0,
    line: { color: C.accent, width: 1 },
  });

  s.addText([
    { text: "*** p < 0.001", options: { bold: true, color: C.warn } },
    { text: "    n.s. = 不显著", options: { color: C.textLight } },
    { text: "    |d| > 0.8 = 大效应    |d| > 0.5 = 中效应    |d| < 0.2 = 微小", options: { color: C.textLight } },
  ], {
    x: 0.6, y: 4.1, w: 8.8, h: 0.25,
    fontSize: 9, fontFace: FONT, margin: 0,
  });

  const conclusions = [
    "除「Avg vs 初始策略」外，所有配对差异高度显著 (p < 0.001)",
    "Pool vs Avg 效应量最大 (d=−3.78) — 两种方法结果差异极大",
    "Avg vs 初始策略 d=−0.24，效应微小且不显著 — 课程学习已提供强基线，博弈训练增量有限",
  ];
  conclusions.forEach((c, i) => {
    s.addText([
      { text: `▸ `, options: { color: C.accent } },
      { text: c, options: {} },
    ], {
      x: 0.6, y: 4.45 + i * 0.24, w: 8.8, h: 0.22,
      fontSize: 10, fontFace: FONT, color: C.text, margin: 0,
    });
  });

  addSlideNum(s, 15, TOTAL);
}

// ============================================================
// Slide 16: Discussion - Why Pool Failed
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "讨论 — 对手池博弈表现差劲的原因分析");

  // Three reasons
  const reasons = [
    {
      title: "策略循环的脆弱性",
      desc: "4人混战 ≠ 两人零和\n没有单一纳什均衡\n策略循环不断\nPFSP 聚焦困难对手\n→ 反复适应被淘汰策略",
    },
    {
      title: "多人博弈的采样偏差",
      desc: "PFSP 采样关注 \"智能体0\" 胜率\n但4人博弈中，胜率包含\n其他智能体策略交互效应\n聚焦低胜率对手 ≠ 有价值训练\n→ 采样信号被污染",
    },
    {
      title: "博弈训练的边际收益递减",
      desc: "初始课程策略已 603.1\n所有博弈训练均未能超越\n近战对称风险 + 资源稳定收益\n→ \"避免对抗、稳定拾球\"\n仍是强竞争策略",
    },
  ];

  reasons.forEach((r, i) => {
    const x = 0.6 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.1, w: 2.8, h: 3.0,
      fill: { color: C.white },
      line: { color: C.warn, width: 1.2 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.1, w: 2.8, h: 0.45,
      fill: { color: C.warn, transparency: 30 },
    });
    s.addText(r.title, {
      x: x + 0.1, y: 1.12, w: 2.6, h: 0.4,
      fontSize: 12, fontFace: FONT, color: C.text, bold: true,
      align: "center", margin: 0,
    });
    s.addText(r.desc, {
      x: x + 0.15, y: 1.7, w: 2.5, h: 2.2,
      fontSize: 10.5, fontFace: FONT, color: C.textLight, margin: 0,
    });
  });

  // Summary
  s.addText("平均策略博弈的优势：策略平均天然平滑化 → 不追逐极端 → 在多人策略循环中更鲁棒", {
    x: 0.6, y: 4.35, w: 8.8, h: 0.4,
    fontSize: 12, fontFace: FONT, color: C.accent, bold: true, align: "center", margin: 0,
  });

  s.addShape(pres.shapes.LINE, {
    x: 2.0, y: 4.75, w: 6, h: 0,
    line: { color: C.accent, width: 1 },
  });

  addSlideNum(s, 16, TOTAL);
}

// ============================================================
// Slide 17: Conclusions & Contributions
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSectionTitle(s, "总结与主要贡献");

  // Contributions
  s.addText("主要贡献", {
    x: 0.6, y: 1.0, w: 8.8, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const contribs = [
    { num: "1", title: "完整实验方案", desc: "从环境构建 (Tiny Swords) → 基础训练 (课程学习 + PBRS) → 博弈训练 (IPPO/Pool/Avg) → 多维量化评估 的端到端研究管线" },
    { num: "2", title: "多维度实证对比", desc: "在非对称四人一般和博弈场景下，对三类训练方法进行了具备统计检验 (配对t/F-test) 和效应量 (Cohen's d) 分析的实证比较" },
    { num: "3", title: "环境-方法关系揭示", desc: "初始策略对照揭示了课程学习、博弈训练、环境对抗强度与评估指标之间的复杂关系：强基线 + 对称性 → 博弈训练增量有限" },
    { num: "4", title: "工程经验积累", desc: "Godot RL Agents 集成、观测编码设计 (固定槽位/有效掩码)、奖励归一化 (Welford)、Optuna 超参数搜索 的实践方案" },
  ];

  contribs.forEach((c, i) => {
    const y = 1.4 + i * 0.7;
    // Number
    s.addShape(pres.shapes.OVAL, {
      x: 0.7, y: y, w: 0.45, h: 0.45,
      fill: { color: C.accent },
    });
    s.addText(c.num, {
      x: 0.7, y: y, w: 0.45, h: 0.45,
      fontSize: 14, fontFace: FONT_EN, color: C.white, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
    // Title
    s.addText(c.title, {
      x: 1.3, y: y, w: 1.6, h: 0.25,
      fontSize: 12, fontFace: FONT, color: C.accent, bold: true, margin: 0,
    });
    // Description
    s.addText(c.desc, {
      x: 1.3, y: y + 0.25, w: 7.8, h: 0.4,
      fontSize: 10, fontFace: FONT, color: C.textLight, margin: 0,
    });
    // Divider
    if (i < contribs.length - 1) {
      addLine(s, y + 0.65, C.line, 0.2);
    }
  });

  // Bottom bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 4.6, w: 8.8, h: 0.45,
    fill: { color: C.accentLight },
  });
  s.addText("理论层面 · 工程层面 · 评估层面 — 三者结合的系统性贡献", {
    x: 0.6, y: 4.6, w: 8.8, h: 0.45,
    fontSize: 12, fontFace: FONT, color: C.accent,
    align: "center", valign: "middle", margin: 0,
  });

  addSlideNum(s, 17, TOTAL);
}

// ============================================================
// Slide 18: Future Work & Thanks
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Title accent line
  s.addShape(pres.shapes.LINE, {
    x: 0.6, y: 0.5, w: 8.8, h: 0,
    line: { color: C.accent, width: 3 },
  });

  // Future directions
  s.addText("未来研究方向", {
    x: 0.6, y: 0.8, w: 8.8, h: 0.4,
    fontSize: 16, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const futures = [
    "实现完整联盟训练 (AlphaStar 风格)：Main Agent + Main Exploiter + League Exploiter 三类角色",
    "将平均策略从手工算术平均升级为神经网络逼近 (NFSP)，提升表达能力",
    "丰富策略多样性：引入远程攻击、技能冷却系统、控制效果、地形争夺等新机制",
    "从两人零和博弈 (如石头剪刀布) 开始深入探讨 RL + 博弈论结合的理论保证",
    "扩展为支持任意人数、可配置分组的多人混战模式，增强博弈复杂性",
  ];

  futures.forEach((f, i) => {
    s.addText([
      { text: `0${i + 1}  `, options: { bold: true, color: C.accent, fontSize: 10 } },
      { text: f, options: {} },
    ], {
      x: 0.8, y: 1.3 + i * 0.55, w: 8.4, h: 0.45,
      fontSize: 11, fontFace: FONT, color: C.text, margin: 0,
    });
    addLine(s, 1.7 + i * 0.55, C.line, 0.2);
  });

  // Thanks section
  s.addShape(pres.shapes.LINE, {
    x: 2.0, y: 3.7, w: 6, h: 0,
    line: { color: C.accent, width: 1.5 },
  });

  s.addText("感谢各位老师批评指正", {
    x: 0.6, y: 3.9, w: 8.8, h: 0.7,
    fontSize: 24, fontFace: FONT, color: C.accent, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  // Bottom info
  s.addText("蒋伦吉  ·  华北电力大学  ·  计算机科学与技术  ·  2026 年 6 月", {
    x: 0.6, y: 4.85, w: 8.8, h: 0.35,
    fontSize: 10, fontFace: FONT, color: C.textLight,
    align: "center", margin: 0,
  });

  addSlideNum(s, 18, TOTAL);
}

// ============================================================
// Write file
// ============================================================
const outPath = "D:/schoolTour/softwares/multi-agent-gameplay/答辩幻灯片.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("PPTX saved to: " + outPath);
}).catch(err => {
  console.error("Error:", err);
});
