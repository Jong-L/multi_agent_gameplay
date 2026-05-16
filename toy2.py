import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置Seaborn风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
np.random.seed(42)

# 生成多条曲线数据
def generate_smooth_curve(x, base_func, noise_scale=5):
    """生成平滑曲线数据"""
    y = base_func(x) + np.random.normal(0, noise_scale, len(x))
    # 使用移动平均平滑曲线
    window_size = 5
    kernel = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, kernel, mode='same')
    return y_smooth

# 定义x轴数据
x = np.linspace(0, 800, 200)

# 生成三条不同的曲线
curves = []
labels = ['Algorithm A', 'Algorithm B', 'Algorithm C']

# 曲线1: 指数衰减
curve1_y = generate_smooth_curve(x, lambda x: 200 * np.exp(-0.02 * x) + 20)
curves.append((x, curve1_y))

# 曲线2: 对数衰减
curve2_y = generate_smooth_curve(x, lambda x: 180 / (1 + 0.01 * x) + 30)
curves.append((x, curve2_y))

# 曲线3: 幂律衰减
curve3_y = generate_smooth_curve(x, lambda x: 150 * np.power(x + 1, -0.3) + 40)
curves.append((x, curve3_y))

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 定义渐变色彩
colors = ['#8B00FF', '#1E90FF', '#2E8B57']  # 紫色、蓝色、绿色
alphas = [0.4, 0.4, 0.4]

# 绘制每条曲线及其填充区域
for i, (x_data, y_data) in enumerate(curves):
    # 绘制主曲线
    ax.plot(x_data, y_data, color=colors[i], linewidth=1.5, label=labels[i], zorder=3)
    
    # 添加额外的淡色填充以增强水墨效果
    ax.fill_between(x_data, y_data - 10, y_data + 10, alpha=alphas[i]*0.3, color=colors[i], zorder=1)

# 添加红色水平参考线
ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Baseline')

# 设置坐标轴
ax.set_xlim(0, 800)
ax.set_ylim(0, 220)
ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Steps per Episode', fontsize=12, fontweight='bold')

# 添加标题
ax.set_title('(e) DDQN', fontsize=14, fontweight='bold', pad=15)

# 添加图例
ax.legend(loc='upper right', framealpha=0.8, fontsize=10)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.6)

# 美化边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 调整布局
plt.tight_layout()

# 保存高分辨率图像（可选）
plt.savefig('ddqn_training_curves.png', dpi=300, bbox_inches='tight', facecolor='white')

# 显示图形
plt.show()
