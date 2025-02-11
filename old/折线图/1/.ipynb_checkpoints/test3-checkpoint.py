import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import pandas as pd
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D  # 导入Line2D用于创建图例的圆形标记

# 获取当前目录下所有的 G-YYYY-MM-DD 格式的文件
files = [f for f in os.listdir() if f.startswith('G-') and f.endswith('.csv')]

# 创建一个存储度分布图的目录
output_dir = 'degree'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def plot_degree_distribution(G, file_name, color, line_width=1):
    # 获取度分布数据
    degree_histogram = nx.degree_histogram(G)
    x = range(len(degree_histogram))  # 度值
    y = [z / float(sum(degree_histogram)) for z in degree_histogram]  # 频率

    # 使用样条插值平滑度分布曲线
    x_smooth = np.linspace(min(x), max(x), 300)  # 生成更细的x坐标
    spline = make_interp_spline(x, y, k=3)  # 立方样条插值
    y_smooth = spline(x_smooth)  # 获取平滑后的y坐标

    # 排除负概率值，确保所有概率值大于等于 0
    y_smooth = np.maximum(y_smooth, 0)  # 将负值替换为 0

    # 在同一张图上绘制多个文件的度分布（度值-概率）
    plt.plot(x_smooth, y_smooth, label=file_name, color=color, linewidth=line_width)


# 定义柔和协调的颜色
colors = ['#A3C8FF', '#1E3A8A', '#34D399', '#C084FC']  # 淡蓝、深蓝、绿色、淡紫
# 如果文件数量超过4个，我们将重复使用这些颜色
if len(files) > len(colors):
    colors = colors * (len(files) // len(colors)) + colors[:len(files) % len(colors)]

# 创建一个图表，绘制所有文件的度分布
plt.figure(figsize=(8, 6), dpi=150)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel("Degree", size=14)
plt.ylabel("Frequency", size=14)
plt.xticks(fontproperties='Times New Roman', size=13)
plt.yticks(fontproperties='Times New Roman', size=13)

# 对每个文件进行处理
for idx, file in enumerate(files):
    print(f"Processing file: {file}")

    # 使用 pandas 读取 CSV 数据，处理为邻接矩阵
    df = pd.read_csv(file, header=None, encoding='utf-8')

    # 确保所有数据都转化为数值类型（无法转换的部分会变为 NaN）
    df = df.apply(pd.to_numeric, errors='coerce')

    # 删除节点名为空的行和列（即所有元素都是 NaN 的行和列）
    df = df.dropna(axis=0, how='all')  # 删除全为空的行
    df = df.dropna(axis=1, how='all')  # 删除全为空的列

    # 转换为无向图的邻接矩阵（假设矩阵中值大于 0 且小于等于 1 表示边的权重）
    adj_matrix = df.values
    adj_matrix = np.where((adj_matrix > 0) & (adj_matrix <= 1), 1, 0)  # 1 表示边，0 表示没有边

    # 构建无向图
    G = nx.from_numpy_array(adj_matrix)

    # 调用绘图函数，将不同文件的度分布绘制在同一张图上
    color = colors[idx % len(colors)]  # 根据文件索引选择颜色
    plot_degree_distribution(G, file.split('.')[0], color)

# 添加图例（优化）使用小圆圈
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=file.split('.')[0]) for
    idx, (file, color) in enumerate(zip(files, colors))]

plt.legend(
    handles=legend_elements,  # 使用自定义的图例元素
    loc='upper right',  # 图例位置
    bbox_to_anchor=(1.0, 1),  # 确保图例不会遮挡图形
    frameon=False,  # 不显示图例背景框
    fontsize=12,  # 图例字体大小
    title="Date",  # 图例标题
    title_fontsize=14  # 图例标题字体大小
)

# 添加标题
plt.title('Degree Distribution for Multiple Files', fontsize=14)

# 去掉背景网格线
plt.grid(False)

# 保存图像
plt.savefig(f'{output_dir}/all_files_degree_distribution_smooth_with_legend.png',
            bbox_inches='tight')  # 使用 tight 确保保存时图形完整
plt.close()

print("Degree distribution plots for all files saved on one graph with smooth curves and improved legend.")
