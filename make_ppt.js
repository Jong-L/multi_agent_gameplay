const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "蒋伦吉";
pres.title = "基于机器学习的对抗性机器博弈研究";

// ===== 设计常量 =====
const C = {
  DARK:  "2C3E50",
  ACCENT:"3498DB",
  LBLUE: "5DADE2",
  GRAY:  "7F8C8D",
  LIGHT: "BDC3C7",
  BG:    "F8F9FA",
  WHITE: "FFFFFF",
  RED:   "E74C3C",
  GREEN: "27AE60",
  ORANGE:"E67E22",
};
const IMG = "D:/schoolTour/softwares/multi-agent-gameplay/article/imgs";
const T = 21; // total slides

const makeShadow = () => ({ type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.10 });

function addFooter(slide, text) {
  slide.addText(text, { x: 0.5, y: 5.15, w: 9, h: 0.3, fontSize: 9, color: C.LIGHT, align: "center", fontFace: "Calibri" });
}
function addPageNum(slide, num) {
  slide.addText(`${num} / ${T}`, { x: 9.0, y: 5.2, w: 0.7, h: 0.3, fontSize: 9, color: C.LIGHT, align: "right", fontFace: "Calibri" });
}
function addSectionTitle(slide, section, title) {
  slide.addText(section, { x: 0.6, y: 0.25, w: 8.8, h: 0.35, fontSize: 11, color: C.ACCENT, fontFace: "Arial", bold: true, charSpacing: 3 });
  slide.addShape(pres.shapes.LINE, { x: 0.6, y: 0.62, w: 8.8, h: 0, line: { color: C.LIGHT, width: 0.5 } });
  slide.addText(title, { x: 0.6, y: 0.7, w: 8.8, h: 0.55, fontSize: 24, color: C.DARK, fontFace: "Arial", bold: true, margin: 0 });
}
function hdrCell(t) {
  return { text: t, options: { bold: true, color: C.WHITE, fill: { color: C.LBLUE }, align: "center", fontSize: 9 } };
}
function dataCell(t, opt) {
  return { text: t, options: { align: "center", fontSize: 9, ...opt } };
}

// ===== SLIDE 1: 封面 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  s.addImage({ path: `${IMG}/ncepu.png`, x: 1.5, y: 0.2, w: 7.0, h: 0.9, sizing: { type: "contain", w: 7.0, h: 0.9 } });
  s.addShape(pres.shapes.LINE, { x: 1.2, y: 1.25, w: 0, h: 3.8, line: { color: C.LBLUE, width: 1.5 } });
  s.addText("基于机器学习的对抗性机器博弈研究", {
    x: 1.6, y: 1.9, w: 7.2, h: 0.8, fontSize: 30, color: C.LBLUE, fontFace: "SimSun", bold: true, align: "center", valign: "middle",
  });
  s.addShape(pres.shapes.LINE, { x: 2.5, y: 2.85, w: 5, h: 0, line: { color: C.LBLUE, width: 0.75 } });
  s.addText("华北电力大学本科毕业设计答辩", {
    x: 1.6, y: 3.05, w: 7.2, h: 0.45, fontSize: 15, color: C.GRAY, fontFace: "SimSun", align: "center", valign: "middle",
  });
  s.addText("答辩人：蒋伦吉　|　专业：计算机科学与技术　|　指导教师：刘春阳", {
    x: 1.6, y: 3.85, w: 7.2, h: 0.35, fontSize: 11, color: C.GRAY, fontFace: "Calibri", align: "center", valign: "middle",
  });
  s.addText("2026年6月", {
    x: 1.6, y: 4.3, w: 7.2, h: 0.3, fontSize: 11, color: C.GRAY, fontFace: "Calibri", align: "center", valign: "middle",
  });
}

// ===== SLIDE 2: 目录 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "CONTENTS", "内容概览");
  const items = [
    { num: "01", title: "研究背景与目标", desc: "博弈对抗是AI的试金石，明确研究问题与框架" },
    { num: "02", title: "环境设计与基础训练", desc: "Tiny Swords竞技场构建、奖励工程与消融实验" },
    { num: "03", title: "网络架构设计", desc: "MLP / Segmented MLP / GRU-MLP 对比" },
    { num: "04", title: "多智能体博弈训练", desc: "IPPO / 对手池博弈 / 平均策略博弈 三种方法对比" },
    { num: "05", title: "结论与展望", desc: "核心发现、实验结论与未来方向" },
  ];
  items.forEach((item, i) => {
    const y = 1.4 + i * 0.72;
    s.addText(item.num, { x: 0.8, y, w: 0.6, h: 0.5, fontSize: 22, color: C.ACCENT, fontFace: "Arial", bold: true, valign: "middle" });
    s.addShape(pres.shapes.LINE, { x: 1.5, y: y + 0.08, w: 0.4, h: 0, line: { color: C.LIGHT, width: 0.5 } });
    s.addText(item.title, { x: 2.0, y, w: 5, h: 0.28, fontSize: 16, color: C.DARK, fontFace: "Arial", bold: true, margin: 0 });
    s.addText(item.desc, { x: 2.0, y: y + 0.28, w: 6.5, h: 0.22, fontSize: 11, color: C.GRAY, fontFace: "Calibri", margin: 0 });
    s.addShape(pres.shapes.LINE, { x: 0.8, y: y + 0.58, w: 8.4, h: 0, line: { color: C.BG, width: 0.5 } });
  });
  addPageNum(s, 2);
}

// ===== SLIDE 3: 研究内容与框架 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "FRAMEWORK", "研究内容与框架");
  s.addImage({ path: `${IMG}/research_framework.png`, x: 0.8, y: 1.2, w: 8.4, h: 3.5, sizing: { type: "contain", w: 8.4, h: 3.5 } });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 3);
}

// ===== SLIDE 4: 环境设计 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "ENVIRONMENT", "Tiny Swords — 2D俯视角四人混战竞技场");
  s.addImage({ path: `${IMG}/game_snapshot_0.png`, x: 0.6, y: 1.1, w: 5.2, h: 3.4, sizing: { type: "contain", w: 5.2, h: 3.4 } });
  const cards = [
    { label: "引擎", value: "Godot 4.x + Godot RL Agents" },
    { label: "地图", value: "18×18栅格，封闭竞技场，含障碍物" },
    { label: "角色", value: "4智能体（四角出生，差异化奖励）" },
    { label: "元素", value: "敌人NPC、A/B类奖励球、中央资源区" },
    { label: "视角", value: "2D俯视，视野半径200px部分可观测" },
    { label: "机制", value: "近战扇形攻击、生命值系统、3分钟对局" },
  ];
  cards.forEach((c, i) => {
    const y = 1.2 + i * 0.6;
    s.addText(c.label, { x: 6.0, y, w: 1.2, h: 0.25, fontSize: 10, color: C.ACCENT, fontFace: "Arial", bold: true, margin: 0 });
    s.addText(c.value, { x: 7.2, y, w: 2.5, h: 0.25, fontSize: 10, color: C.DARK, fontFace: "Calibri", margin: 0 });
    if (i < cards.length - 1) {
      s.addShape(pres.shapes.LINE, { x: 6.0, y: y + 0.4, w: 3.7, h: 0, line: { color: C.BG, width: 0.5 } });
    }
  });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 4);
}

// ===== SLIDE 5: 观测空间设计 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "OBSERVATION", "观测空间设计 — 174维分段式编码");
  const tableData = [
    [hdrCell("分段"), hdrCell("维度"), hdrCell("编码方式"), hdrCell("关键设计")],
    ["SELF",  "15",  "1×15", "坐标、血量、上一动作 one-hot、朝向"],
    ["PLAYER","33",  "3×11", "相对位置、血量、有效位掩码、固定槽位"],
    ["BALL",  "40",  "8×5",  "按距离排序、类型、有效位掩码"],
    ["ENEMY", "50",  "5×10", "固定槽位、攻击状态、视野内追踪"],
    ["LiDAR", "36",  "36×1", "36条射线归一化距离 [0,1]"],
  ];
  s.addTable(tableData, {
    x: 0.6, y: 1.3, w: 8.8, colW: [1.2, 0.8, 1.2, 5.6],
    border: { pt: 0.5, color: C.LIGHT },
    rowH: [0.4, 0.45, 0.45, 0.45, 0.45, 0.45],
    fontFace: "Calibri", fontSize: 11, color: C.DARK,
  });
  s.addText([
    { text: "核心设计原则：", options: { bold: true, color: C.ACCENT, breakLine: true } },
    { text: "① 以自身为中心编码（平移不变性）  ② 固定槽位编码（时序一致性）  ③ 有效位掩码（区分零填充与无效信息）  ④ 动力学信息（攻击状态、冷却）", options: {} },
  ], { x: 0.6, y: 4.0, w: 8.8, h: 0.8, fontSize: 11, color: C.DARK, fontFace: "Calibri" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 5);
}

// ===== SLIDE 6: 奖励函数设计（完整表格） =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "REWARD", "奖励函数设计");
  const rows = [
    ["拾取 A 球",       "10",   "6",    "12",   "6"],
    ["拾取 B 球",       "15",   "3",    "10",   "4"],
    ["对敌人造成伤害",    "10",   "8",    "3",    "4.05"],
    ["对玩家造成伤害",    "15",   "10",   "4",    "6.05"],
    ["击杀敌人",        "30",   "16",   "6",    "8"],
    ["击杀玩家",        "45",   "20",   "8",    "12"],
    ["受到伤害",        "−10",  "−5",   "−10",  "−8"],
    ["死亡",           "−15",  "−10",  "−20",  "−15"],
    ["攻击动作（/次）",   "−0.02","−0.01","−0.06","−0.05"],
    ["移动（/通信帧）",   "−0.001","−0.001","−0.001","−0.001"],
    ["静止（/通信帧）",   "−0.005","−0.005","−0.005","−0.005"],
    ["中心区域奖励（/帧）","0.003","0.003","0.003","0.003"],
    ["撞墙惩罚",        "−0.5", "−0.5", "−0.5", "−0.5"],
  ];
  const tableData = [
    [hdrCell("奖励事件"), hdrCell("默认值"), hdrCell("战士型(0,2)"), hdrCell("采集型(1)"), hdrCell("均衡型(3)")],
    ...rows.map(r => r.map((v, ci) => {
      if (ci === 0) return { text: v, options: { fontSize: 8.5, align: "left" } };
      return { text: v, options: { align: "center", fontSize: 8.5 } };
    })),
  ];
  s.addTable(tableData, {
    x: 0.6, y: 1.2, w: 8.8, colW: [3.2, 1.4, 1.4, 1.4, 1.4],
    border: { pt: 0.5, color: C.LIGHT },
    rowH: [0.32, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24],
    fontFace: "Calibri",
  });
  s.addText("稠密塑形：PBRS 奖励球势能引导 + 中心区域稠密奖励 + 移动/待机/攻击微惩罚（中心区域待机惩罚豁免）", {
    x: 0.6, y: 4.85, w: 8.8, h: 0.3, fontSize: 10, color: C.GRAY, fontFace: "Calibri",
  });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 6);
}

// ===== SLIDE 7: 有效位掩码消融 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "ABLATION: MASK", "消融实验 — 有效位掩码");
  s.addText("区分\"数值为0的有效观测\"与\"实体不可见的零填充\"", {
    x: 0.6, y: 1.15, w: 8.8, h: 0.3, fontSize: 12, color: C.GRAY, fontFace: "Calibri",
  });
  s.addImage({ path: `${IMG}/valid_mask_comparison_player_ball_score.png`, x: 0.4, y: 1.5, w: 4.3, h: 2.2, sizing: { type: "contain", w: 4.3, h: 2.2 } });
  s.addImage({ path: `${IMG}/valid_mask_comparison_total_ball_score.png`, x: 5.1, y: 1.5, w: 4.3, h: 2.2, sizing: { type: "contain", w: 4.3, h: 2.2 } });
  s.addText([
    { text: "结论：", options: { bold: true, color: C.ACCENT } },
    { text: "有效位掩码使网络能明确区分有效信息与填充噪声，提升后期训练质量。未使用掩码时，智能体难以区分视野外实体与零值有效数据，导致学习信号混乱。" },
  ], { x: 0.6, y: 3.9, w: 8.8, h: 0.6, fontSize: 11, color: C.DARK, fontFace: "Calibri" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 7);
}

// ===== SLIDE 8: 奖励塑形方法（新插入，介于有效位掩码与PBRS结果之间） =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "ABLATION: PBRS METHODS", "奖励塑形方法 — 势函数与稠密引导");
  const methods = [
    { name: "稀疏奖励", formula: "仅在拾球时获得 R，中间无引导信号", note: "基线方案，学习速度最慢" },
    { name: "线性势函数", formula: "Φ(s) = −(R / rᵥ) · d + R", note: "距球越近奖励越高，线性衰减" },
    { name: "反比例势函数", formula: "Φ(s) = R · rᵥ / (d + rᵥ)", note: "近距离梯度大，远距离趋于平缓" },
    { name: "指数势函数", formula: "Φ(s) = R · exp(−d / rᵥ)", note: "近距离梯度大于线性，远距离快速衰减" },
    { name: "距离奖励", formula: "Φ(s) = d_t − d_{t+1}", note: "不需球分值R，对所有球统一施加" },
  ];
  methods.forEach((m, i) => {
    const y = 1.25 + i * 0.85;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 0.07, h: 0.65, fill: { color: C.LBLUE } });
    s.addText(m.name, { x: 0.85, y, w: 1.8, h: 0.25, fontSize: 12, color: C.DARK, fontFace: "Arial", bold: true, margin: 0 });
    s.addText(m.formula, { x: 0.85, y: y + 0.25, w: 4.5, h: 0.2, fontSize: 10, color: C.ACCENT, fontFace: "Calibri", margin: 0, italic: true });
    s.addText(m.note, { x: 0.85, y: y + 0.45, w: 5.0, h: 0.2, fontSize: 9, color: C.GRAY, fontFace: "Calibri", margin: 0 });
  });
  // 右侧：最近球 vs 所有球
  s.addShape(pres.shapes.LINE, { x: 5.5, y: 1.3, w: 0, h: 3.8, line: { color: C.LIGHT, width: 0.5, dashType: "dash" } });
  s.addText("多球聚合策略", {
    x: 5.8, y: 1.25, w: 3.5, h: 0.3, fontSize: 12, color: C.ACCENT, fontFace: "Arial", bold: true, margin: 0,
  });
  s.addText([
    { text: "最近球模式", options: { bold: true, color: C.DARK, breakLine: true } },
    { text: "仅对视野内最近的奖励球计算势能，将多目标导航分解为阶段性子目标", options: { fontSize: 10, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "所有球模式", options: { bold: true, color: C.DARK, breakLine: true } },
    { text: "对所有可见球叠加势能，但球间势能可能相互抵消，稀释引导信号", options: { fontSize: 10 } },
  ], { x: 5.8, y: 1.55, w: 3.5, h: 3.0, fontSize: 10, color: C.GRAY, fontFace: "Calibri", valign: "top" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 8);
}

// ===== SLIDE 9: 奖励球势能塑形 — 最近球 vs 所有球（上） =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "ABLATION: PBRS RESULTS", "消融实验 — 奖励球势能塑形对比结果");
  s.addImage({ path: `${IMG}/per_player_comparison-ball.png`, x: 0.4, y: 1.3, w: 4.5, h: 3.3, sizing: { type: "contain", w: 4.5, h: 3.3 } });
  s.addImage({ path: `${IMG}/total_comparison-ball.png`, x: 5.1, y: 1.3, w: 4.5, h: 3.3, sizing: { type: "contain", w: 4.5, h: 3.3 } });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 9);
}

// ===== SLIDE 10: 奖励球势能塑形 — 势函数对比（下） =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "ABLATION: PBRS RESULTS", "消融实验 — 奖励球势能塑形对比结果（续）");
  s.addImage({ path: `${IMG}/nearest_vs_all_player_comparison.png`, x: 0.4, y: 1.3, w: 4.5, h: 3.3, sizing: { type: "contain", w: 4.5, h: 3.3 } });
  s.addImage({ path: `${IMG}/nearest_vs_all_total_comparison.png`, x: 5.1, y: 1.3, w: 4.5, h: 3.3, sizing: { type: "contain", w: 4.5, h: 3.3 } });
  addFooter(s, "结论：线性势函数 + 最近球模式最优，多球叠加导致势能抵消");
  addPageNum(s, 10);
}

// ===== SLIDE 11: 避障方法对比 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "ABLATION: WALL", "避障方法对比");
  s.addImage({ path: `${IMG}/wall_potential_comparison_total_score.png`, x: 0.5, y: 1.3, w: 4.3, h: 2.8, sizing: { type: "contain", w: 4.3, h: 2.8 } });
  s.addImage({ path: `${IMG}/wall_potential_comparison_total_walls.png`, x: 5.2, y: 1.3, w: 4.3, h: 2.8, sizing: { type: "contain", w: 4.3, h: 2.8 } });
  s.addText([
    { text: "结论：", options: { bold: true, color: C.ACCENT } },
    { text: "稠密塑形虽大幅降低撞墙，但导致智能体系统性回避障碍物附近的奖励球——\"不是越远越好\"。" },
  ], { x: 0.8, y: 4.3, w: 8.4, h: 0.6, fontSize: 12, color: C.DARK, fontFace: "Calibri", valign: "middle" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 11);
}

// ===== SLIDE 12: 网络架构设计 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "NETWORK", "网络架构设计");
  s.addImage({ path: `${IMG}/network_architectures.png`, x: 0.6, y: 1.1, w: 8.8, h: 3.5, sizing: { type: "contain", w: 8.8, h: 3.5 } });
  s.addText([
    { text: "MLP：", options: { bold: true, color: C.ACCENT } },
    { text: "观测直入全连接   ", options: {} },
    { text: "Segmented MLP：", options: { bold: true, color: C.ACCENT } },
    { text: "各语义段独立子MLP→融合   ", options: {} },
    { text: "GRU-MLP：", options: { bold: true, color: C.ACCENT } },
    { text: "时序段经GRU捕获时间依赖，BALL段独立MLP", options: {} },
  ], { x: 0.6, y: 4.6, w: 8.8, h: 0.4, fontSize: 11, color: C.DARK, fontFace: "Calibri" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 12);
}

// ===== SLIDE 13: 网络架构对比 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "NETWORK COMPARISON", "网络架构对比");
  // S1 场景对比
  s.addImage({ path: `${IMG}/s1_total_ball_score.png`, x: 0.4, y: 1.15, w: 4.3, h: 1.9, sizing: { type: "contain", w: 4.3, h: 1.9 } });
  // S4 场景对比
  s.addImage({ path: `${IMG}/sarl_total_score.png`, x: 5.1, y: 1.15, w: 4.3, h: 1.9, sizing: { type: "contain", w: 4.3, h: 1.9 } });
  // 超参数搜索奖励分布
  s.addImage({ path: `${IMG}/optuna_round2_compare.png`, x: 0.6, y: 3.15, w: 8.8, h: 1.9, sizing: { type: "contain", w: 8.8, h: 1.9 } });
  addFooter(s, "结论：MLP得分最高→采用为基础架构；GRU时序优势未转化为收益；SegMLP有优化潜力");
  addPageNum(s, 13);
}

// ===== SLIDE 14: 博弈方法评估与对比 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "METHODS", "博弈方法评估与对比");
  // 背景说明
  s.addText("构造一个包含20种策略的对手池，使用博弈训练方法，在多智能体场景下，以智能体0为训练对象评估不同方法的训练效果",
    { x: 0.6, y: 0.95, w: 8.8, h: 0.45, fontSize: 12, color: C.DARK, fontFace: "Calibri", italic: true, valign: "middle" });
  // 副标题
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.5, w: 0.06, h: 0.3, fill: { color: C.ACCENT } });
  s.addText("三种博弈训练方法", { x: 0.85, y: 1.45, w: 8.5, h: 0.35, fontSize: 15, color: C.DARK, fontFace: "Arial", bold: true, margin: 0 });
  const methods = [
    {
      name: "独立近端策略优化（IPPO）", color: C.ACCENT,
      desc: "四个智能体同步IPPO训练，将其他智能体视为环境的一部分，不引入任何集中式组件",
    },
    {
      name: "对手池博弈 (PFSP)", color: C.ORANGE,
      desc: "基于PSRO框架，PFSP采样对手组合，聚焦困难对手进行针对性训练",
    },
    {
      name: "平均策略博弈", color: C.GREEN,
      desc: "基于虚拟博弈思想，将20组对手策略在动作概率空间算术平均作为对手",
    },
  ];
  methods.forEach((m, i) => {
    const y = 1.95 + i * 1.0;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 0.08, h: 0.85, fill: { color: m.color } });
    s.addText(m.name, { x: 0.9, y, w: 2.2, h: 0.35, fontSize: 14, color: m.color, fontFace: "Arial", bold: true, margin: 0 });
    s.addText(m.desc, { x: 0.9, y: y + 0.38, w: 8.3, h: 0.35, fontSize: 12, color: C.DARK, fontFace: "Calibri", margin: 0 });
  });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 14);
}

// ===== SLIDE 15: 多智能体评估流程 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "PIPELINE", "多智能体博弈 — 评估流程");
  s.addImage({ path: `${IMG}/gameplay_evaluate.png`, x: 0.5, y: 1.1, w: 9.0, h: 4.0, sizing: { type: "contain", w: 9.0, h: 4.0 } });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 15);
}

// ===== SLIDE 16: 多智能体结果 — 奖励对比 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "RESULTS: REWARD", "多智能体实验结果 — 奖励对比");
  s.addImage({ path: `${IMG}/eval_compare_reward.png`, x: 0.4, y: 1.1, w: 5.0, h: 2.6, sizing: { type: "contain", w: 5.0, h: 2.6 } });
  s.addImage({ path: `${IMG}/eval_compare_effectsize.png`, x: 5.5, y: 1.1, w: 4.2, h: 2.6, sizing: { type: "contain", w: 4.2, h: 2.6 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.9, w: 8.8, h: 1.2, fill: { color: C.BG } });
  s.addText([
    { text: "核心发现：", options: { bold: true, color: C.RED, breakLine: true } },
    { text: "初始策略对照 (598.5) > 平均策略博弈 (573.2) > IPPO (534.4) > 对手池博弈 (328.2)", options: { bold: true, breakLine: true } },
    { text: "全部六对比较均在 p<0.01 水平显著，|d| 全部 >0.8", options: { breakLine: true } },
    { text: "三种博弈训练方法均未能超越初始策略——\"" },
    { text: "博弈训练导致性能退化", options: { bold: true, color: C.RED } },
    { text: "\"", options: {} },
  ], { x: 0.8, y: 3.95, w: 8.4, h: 1.1, fontSize: 12, color: C.DARK, fontFace: "Calibri" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 16);
}

// ===== SLIDE 17: 多智能体结果 — 行为分析 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "RESULTS: BEHAVIOR", "多智能体实验结果 — 行为策略画像");
  s.addImage({ path: `${IMG}/eval_compare_behavior_events.png`, x: 0.3, y: 1.1, w: 6.2, h: 2.5, sizing: { type: "contain", w: 6.2, h: 2.5 } });
  s.addImage({ path: `${IMG}/eval_compare_behavior_profile.png`, x: 6.6, y: 1.1, w: 3.1, h: 2.5, sizing: { type: "contain", w: 3.1, h: 2.5 } });
  s.addImage({ path: `${IMG}/eval_compare_combat_collection.png`, x: 0.4, y: 3.6, w: 4.3, h: 1.5, sizing: { type: "contain", w: 4.3, h: 1.5 } });
  s.addText([
    { text: "退化机制：", options: { bold: true, color: C.ACCENT, breakLine: true } },
    { text: "博弈训练 → 攻击频率↓、攻击敌人次数↓ → PvE收入↓", options: { breakLine: true, fontSize: 10 } },
    { text: "博弈训练 → 资源收集↑ → 但不足以补偿战斗损失", options: { breakLine: true, fontSize: 10 } },
    { text: "PvP指标极低：每局击杀玩家≈0，四种策略均无有效PvP行为", options: { breakLine: true, fontSize: 10 } },
    { text: "对手越稳定 → 退化越轻（平均>IPPO>PFSP）", options: { breakLine: true, fontSize: 10 } },
  ], { x: 5.0, y: 3.6, w: 4.5, h: 1.5, fontSize: 11, color: C.DARK, fontFace: "Calibri", valign: "middle" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 17);
}

// ===== SLIDE 18: PvP→PvE 迁移实验 — 设计与训练 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "MIGRATION", "PvP→PvE 迁移实验 — 设计与训练");
  // 左侧：训练曲线（放大）
  s.addImage({ path: `${IMG}/ch5_pvp_training_curve.png`, x: 0.4, y: 1.2, w: 4.5, h: 3.5, sizing: { type: "contain", w: 4.5, h: 3.5 } });
  // 右侧：实验设计说明
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.2, w: 4.3, h: 3.5, fill: { color: C.BG } });
  s.addText([
    { text: "实验设计", options: { bold: true, fontSize: 14, color: C.ACCENT, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "目标：检验PvP行为缺失是能力限制还是环境奖励结构驱动？", options: { breakLine: true, fontSize: 12 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "方法：", options: { bold: true, breakLine: true, fontSize: 12 } },
    { text: "① 在纯博弈环境（无敌人/无球）训练PvP智能体", options: { breakLine: true, fontSize: 11 } },
    { text: "② 将训练好的策略迁移至完整环境（含敌人+球）", options: { breakLine: true, fontSize: 11 } },
    { text: "③ 比较冻结策略 vs 继续训练的策略表现差异", options: { fontSize: 11 } },
  ], { x: 5.4, y: 1.3, w: 3.9, h: 3.3, fontFace: "Calibri", color: C.DARK, valign: "top" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 18);
}

// ===== SLIDE 19: PvP→PvE 迁移实验 — 训练过程 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "MIGRATION", "PvP→PvE 迁移实验 — 训练过程");
  s.addImage({ path: `${IMG}/migration_pvp_to_pve_unfrozen.png`, x: 0.6, y: 1.2, w: 8.8, h: 3.8, sizing: { type: "contain", w: 8.8, h: 3.8 } });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 19);
}

// ===== SLIDE 20: PvP→PvE 迁移实验 — 结果对比 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  addSectionTitle(s, "MIGRATION (2/2)", "PvP→PvE 迁移实验 — 冻结 vs 继续训练");
  s.addImage({ path: `${IMG}/migration_pvp_to_pve_frozen_comparison.png`, x: 0.5, y: 1.2, w: 9.0, h: 2.8, sizing: { type: "contain", w: 9.0, h: 2.8 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.1, w: 8.8, h: 1.0, fill: { color: C.BG } });
  s.addText([
    { text: "因果结论：", options: { bold: true, color: C.ACCENT, breakLine: true } },
    { text: "纯博弈环境训练的智能体具备充分的 PvP 能力，但迁移至完整环境后自发将攻击目标从玩家重新分配至敌人。", options: { breakLine: true } },
    { text: "相比于冻结策略，继续训练的策略 PvP 伤害次数下降 61%，PvE 伤害次数上升 23%。", options: { breakLine: true } },
    { text: "→ PvP 行为的缺失是环境奖励结构驱动的理性选择，而非智能体能力限制。", options: { bold: true, color: C.RED } },
  ], { x: 0.8, y: 4.15, w: 8.4, h: 0.9, fontSize: 11, color: C.DARK, fontFace: "Calibri" });
  addFooter(s, "华北电力大学 本科毕业设计答辩");
  addPageNum(s, 20);
}

// ===== SLIDE 21: 致谢 =====
{
  const s = pres.addSlide();
  s.background = { color: C.WHITE };
  s.addShape(pres.shapes.LINE, { x: 0, y: 0.08, w: 10, h: 0, line: { color: C.LBLUE, width: 1.5 } });
  s.addImage({ path: `${IMG}/ncepu.png`, x: 1.5, y: 0.2, w: 7.0, h: 0.9, sizing: { type: "contain", w: 7.0, h: 0.9 } });
  s.addShape(pres.shapes.LINE, { x: 3.2, y: 1.45, w: 3.6, h: 0, line: { color: C.LBLUE, width: 1 } });
  s.addText("谢谢！", {
    x: 1.8, y: 1.6, w: 6.4, h: 1.0, fontSize: 40, color: C.LBLUE, fontFace: "SimSun", bold: true, align: "center", valign: "middle",
  });
  s.addText("敬请各位老师批评指正", {
    x: 1.8, y: 2.65, w: 6.4, h: 0.4, fontSize: 14, color: C.GRAY, fontFace: "SimSun", align: "center", valign: "middle",
  });
  s.addShape(pres.shapes.LINE, { x: 3.2, y: 3.25, w: 3.6, h: 0, line: { color: C.LBLUE, width: 1 } });
  s.addText("答辩人：蒋伦吉　|　指导教师：刘春阳　|　华北电力大学", {
    x: 1.8, y: 3.55, w: 6.4, h: 0.3, fontSize: 10, color: C.GRAY, fontFace: "Calibri", align: "center", valign: "middle",
  });
  s.addShape(pres.shapes.LINE, { x: 0, y: 5.42, w: 10, h: 0, line: { color: C.LBLUE, width: 1.5 } });
}

// ===== 保存 =====
pres.writeFile({ fileName: "D:/schoolTour/softwares/multi-agent-gameplay/答辩PPT-蒋伦吉.pptx" })
  .then(() => console.log("PPT生成成功!"))
  .catch(err => console.error("生成失败:", err));
