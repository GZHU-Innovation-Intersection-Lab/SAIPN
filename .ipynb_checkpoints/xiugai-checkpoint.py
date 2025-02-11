import pandas as pd

# 读取 top5_yinshi.csv 和 top5_xianshi.csv
df_yinshi = pd.read_csv('top10_percent_yinshi.csv')
df_xianshi = pd.read_csv('top10_percent_xianshi.csv')

# 读取 edge_log.csv 文件，存储每一条边的连接信息
edge_log_df = pd.read_csv('edge_log.csv')

# 用于存储符合条件的节点
matched_nodes = []

# 计数器：统计隐式网络比显式网络快的频次
implicit_faster_count = 0
total_check_count = 0  # 用于计算比例

# 获取隐式网络节点的总数
total_yinshi_nodes = df_yinshi.shape[0]

# 获取显式网络节点的总数
total_xianshi_nodes = df_xianshi.shape[0]

# 遍历 top5_yinshi.csv，检查每个隐式网络节点是否符合条件
for index, row in df_yinshi.iterrows():
    w_value_yinshi = row['w_value']
    top5_nodes_yinshi = eval(row['top10_percent_nodes'])  # 解析节点列表

    # 打印当前隐式网络节点的处理进度
    print(f"Processing implicit network node {index + 1}/{total_yinshi_nodes}...")

    # 遍历 top5_yinshi.csv 中的每个隐式网络节点
    for node_id, _ in top5_nodes_yinshi:
        # 对于每个隐式网络节点，遍历 top5_xianshi.csv 中的每个显式网络节点进行匹配检查
        for x_index, row_xianshi in df_xianshi.iterrows():
            w_value_xianshi = row_xianshi['w_value']
            top5_nodes_xianshi = eval(row_xianshi['top10_percent_nodes'])  # 解析节点列表

            # 打印当前显式网络节点的处理进度
            print(f"  Checking explicit network node {x_index + 1}/{total_xianshi_nodes}...")

            total_check_count += 1  # 每检查一个节点，计数增加

            # 遍历显式网络节点并进行时间比较
            for x_node_id, _ in top5_nodes_xianshi:
                found_earlier = False
                found_later = False

                if node_id == x_node_id:  # 如果是同一个节点进行比较
                    if w_value_yinshi < w_value_xianshi:  # 隐式网络节点比显式网络节点时间早
                        found_earlier = True
                    elif w_value_yinshi > w_value_xianshi:  # 隐式网络节点比显式网络节点时间晚
                        found_later = True

                    # 记录符合条件的节点
                    if not (found_earlier and found_later):  # 确保不同时发现比隐式网络节点时间早和晚的情况
                        if found_earlier:
                            matched_nodes.append({
                                'yinshi_node_id': node_id,
                                'yinshi_w_value': w_value_yinshi,
                                'xianshi_node_id': x_node_id,
                                'xianshi_w_value': w_value_xianshi
                            })
                            implicit_faster_count += 1
                        # 如果该节点没有在 top5_xianshi.csv 中直接匹配到，检查 edge_log.csv
                        elif not found_earlier and not found_later:
                            related_nodes = edge_log_df[
                                (edge_log_df['Target_Node'] == x_node_id)]  # 确保显式网络节点是目标节点
                            for _, edge_row in related_nodes.iterrows():
                                # 取得与该节点相关的另一节点的ID（隐式网络节点为源节点）
                                related_node_id = edge_row['Source_Node']
                                if related_node_id == node_id and related_node_id in df_yinshi['top10_percent_nodes'].values:
                                    # 如果相关节点在隐式网络中存在，且符合条件
                                    w_value_yinshi = df_yinshi.loc[df_yinshi['top10_percent_nodes'].str.contains(related_node_id), 'w_value'].values[0]
                                    if w_value_yinshi < w_value_xianshi:  # 隐式网络节点时间更早
                                        matched_nodes.append({
                                            'yinshi_node_id': related_node_id,
                                            'yinshi_w_value': w_value_yinshi,
                                            'xianshi_node_id': x_node_id,
                                            'xianshi_w_value': w_value_xianshi
                                        })
                                        implicit_faster_count += 1
                                        break  # 只处理第一次匹配，跳出相关节点循环

# 将符合条件的节点存储到一个 DataFrame 中
matched_df = pd.DataFrame(matched_nodes)

# 计算比例
implicit_faster_ratio = implicit_faster_count / total_check_count if total_check_count > 0 else 0

# 保存符合条件的节点到 CSV 文件
matched_df.to_csv('matched_nodes_implicit_faster2.csv', index=False, encoding='utf-8')

# 打印结果
print(f"符合条件的节点已保存到 'matched_nodes_implicit_faster2.csv'！")
print(f"隐式网络比显式网络快的频次: {implicit_faster_count}")
print(f"隐式网络比显式网络快的比例: {implicit_faster_ratio:.2%}")

