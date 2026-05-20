# 论文级画图提示词

> **配合 `data_analyze/plot_template.py` 使用。**
> 复制下面这段，贴到对话里即可。

---

**请帮我画一张论文级图表，使用 `data_analyze/publication_plot_utils.py` 的绘图管线（高斯滤波 → 样条插值 → fill_between 淡色误差带）。**

具体要求：
- 数据来源：[描述你的数据格式/路径/列名]
- 图表类型：[单图 / 2×2 子图 / 自定义布局]
- 曲线数量：[N 条方案对比 / 单条训练曲线]
- X 轴：[Episode / Timestep / ...]
- Y 轴：[Average Reward / Loss / ...]
- 图标题：[...]
- 保存路径：[...]

**风格要求（必须遵守，这些已在 utils 中实现，直接调用即可）：**
1. `setup_style()` — seaborn whitegrid + paper context
2. `prepare_curve()` — 高斯滤波去噪 + B-spline 光滑化
3. `plot_with_fill()` — fill_between(alpha=0.18, edgecolor=none) + sns.lineplot(linewidth=2.0)
4. `style_axes()` — 统一轴标签、标题、图例样式
5. `save_figure()` — 300 DPI PNG + bbox_inches=tight
6. 配色用 `COLORS` 字典中的 Wong 2011 色盲友好配色
7. 如有多条曲线，给 legend 内边框加 `edgecolor='#cccccc'`
8. 统计信息框用 `add_stats_box()` 白色圆角半透明

如果数据无 SEM，用固定半带（ymax 的 ~3%）代替误差带或者自行计算。
