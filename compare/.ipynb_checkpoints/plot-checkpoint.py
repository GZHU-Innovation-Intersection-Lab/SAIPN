import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
# matplotlib其实是不支持显示中文的 显示中文需要一行代码设置字体


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体作为默认字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 下面是你的绘图代码...
# 读取Excel文件
df = pd.read_excel('matched_nodes_implicit_faster2.xlsx')

# 将时间列转换为日期格式
df['yinshi_w_value'] = pd.to_datetime(df['yinshi_w_value'])
df['xianshi_w_value'] = pd.to_datetime(df['xianshi_w_value'])

# 调整数据结构以适应绘图要求
df_yinshi = df[['yinshi_node_id', 'yinshi_w_value']].rename(columns={'yinshi_node_id': 'node_id', 'yinshi_w_value': 'date'})
df_xianshi = df[['xianshi_node_id', 'xianshi_w_value']].rename(columns={'xianshi_node_id': 'node_id', 'xianshi_w_value': 'date'})

# 添加一个标记列来区分隐式和显式网络
df_yinshi['network_type'] = '隐式网络'
df_xianshi['network_type'] = '显式网络'

# 合并两个DataFrame
df_combined = pd.concat([df_yinshi, df_xianshi])

# 设置Seaborn样式
sns.set(style="whitegrid")

# 创建图形
plt.figure(figsize=(15, 8))
ax = sns.pointplot(x=df_combined.index, y='date', hue='network_type', data=df_combined,
                   palette={"隐式网络": "blue", "显式网络": "orange"}, markers=["o", "x"], linestyles=["-", "--"])

# 设置标题和标签
ax.set_title('隐式网络与显式网络节点发现时间对比')
ax.set_ylabel('发现时间')
ax.set_xlabel('节点索引')

# 显示图形
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()