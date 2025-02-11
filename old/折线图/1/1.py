import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置 Seaborn 风格
sns.set(style="whitegrid", palette="muted")

# 数据
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图表
plt.figure(figsize=(8, 6))

# 绘制折线图，不显示节点
plt.plot(x, y1, label='sin(x)', linewidth=2, color='b', linestyle='-')
plt.plot(x, y2, label='cos(x)', linewidth=2, color='r', linestyle='--')

# 添加标题和标签
plt.title('Trigonometric Functions', fontsize=16, weight='bold')
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)

# 设置坐标轴范围
plt.xlim([0, 10])
plt.ylim([-1.5, 1.5])

# 显示图例
plt.legend(loc='upper right', fontsize=12)

# 显示网格线
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()
