import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

# Also set font.family explicitly
plt.rcParams['font.family'] = 'sans-serif'

print("Sans-serif fonts:", plt.rcParams['font.sans-serif'])
print("Font family:", plt.rcParams['font.family'])

fig, ax = plt.subplots()
ax.set_title('测试中文渲染')
ax.set_xlabel('回合')
ax.set_ylabel('智能体得分')
fig.savefig('test_cjk.png', dpi=100)
print("Saved test_cjk.png successfully")
