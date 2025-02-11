import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用支持Unicode的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义13种颜色和社区名称
colors = [
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#009E73', '#56B4E9',
    '#E69F00', '#F0E442', '#0072B2', '#999999', '#D55E00', '#56B4E9', '#E69F00'
]
community_names = [
    'Community 1', 'Community 2', 'Community 3', 'Community 4', 'Community 5', 'Community 6',
    'Community 7', 'Community 8', 'Community 9', 'Community 10', 'Community 11', 'Community 12', 'Other Communities'
]

# 创建一个图例
fig, ax = plt.subplots(figsize=(12, 2))  # 调整图例的大小以适应13个社区

# 创建小圆圈
legend_elements = [Line2D([0], [0], marker='o', color='w', label=name,
                          markerfacecolor=color, markersize=10) for color, name in zip(colors, community_names)]

# 绘制图例
ax.legend(handles=legend_elements, loc='center', ncol=13, fontsize=10, frameon=False)

# 调整图例布局
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')  # 关闭坐标轴

# 保存图例为图片
plt.savefig('community_legend_horizontal.png', bbox_inches='tight', pad_inches=0)

# 显示图例（可选，如果你想查看效果）
plt.show()