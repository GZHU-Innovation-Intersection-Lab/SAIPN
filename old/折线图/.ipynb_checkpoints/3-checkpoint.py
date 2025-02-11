import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.arange(0, 10, 1)
y = np.exp(-x / 3)
error = 0.1 + 0.1 * np.sqrt(x)

# 创建图表
plt.figure(figsize=(8, 6))

# 绘制带误差条的折线图，不显示节点
plt.errorbar(x, y, yerr=error, fmt='-', color='darkorange', label='Exp Decay',
             ecolor='skyblue', elinewidth=2, capsize=4, capthick=2)

# 添加标题和标签
plt.title('Exponential Decay with Error Bars', fontsize=16, weight='bold')
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)

# 设置坐标轴范围
plt.xlim([0, 10])
plt.ylim([0, 1.2])

# 显示图例
plt.legend(loc='upper right', fontsize=12)
plt.show()
