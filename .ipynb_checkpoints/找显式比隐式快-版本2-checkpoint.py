import pandas as pd

# 读取 top5_yinshi.csv 和 top5_xianshi.csv
df_xianshi =pd.read_csv('top10_percent_xianshi.csv')
df_yinshi = pd.read_csv('top10_percent_yinshi.csv')

xianshi_dict = {}

# 遍历 top5_xianshi.csv，将节点 ID 和对应的 w_value 存储到字典中
for index, row in df_xianshi.iterrows():
    w_value = row['w_value']
    top5_nodes = eval(row['top10_percent_nodes'])  # 解析节点列表
    for node_id, _ in top5_nodes:
        if node_id not in xianshi_dict:
            xianshi_dict[node_id] = w_value

# 读取 edge_log.csv 文件，存储每一条边的连接信息
edge_log_df = pd.read_csv('edge_log.csv')

# 用于存储符合条件的节点
matched_nodes = []

# 计数器：统计显式网络比隐式网络快的频次
explicit_faster_count = 0
total_check_count = 0  # 用于计算比例

# 遍历 top5_yinshi.csv，检查每个节点是否存在于 xianshi_dict 且时间更晚，或者在 edge_log.csv 中有连接关系
for index, row in df_yinshi.iterrows():
    w_value_yinshi = row['w_value']
    top5_nodes = eval(row['top10_percent_nodes'])  # 解析节点列表
    for node_id, _ in top5_nodes:
        total_check_count += 1  # 每检查一个节点，计数增加

        # 用来记录是否找到显式网络节点的时间既比隐式网络早又比隐式网络晚
        found_earlier = False
        found_later = False

        # 如果该节点在隐式网络中
        if node_id in xianshi_dict:
            w_value_xianshi = xianshi_dict[node_id]
            if w_value_yinshi < w_value_xianshi:  # 判断xian式网络节点是否更晚
                found_later = True
            elif w_value_yinshi > w_value_xianshi:  # 判断显式网络节点是否更早
                found_earlier = True

        # 如果没有同时找到比显式网络节点早和晚的情况，则继续检查
        if not (found_earlier and found_later):
            # 如果找到显式网络节点比隐式网络节点早的情况
            if found_earlier:
                matched_nodes.append({
                    'xianshi_node_id': node_id,
                    'xianshi_w_value': w_value_xianshi,
                    'yinshi_node_id': node_id,
                    'yinshi_w_value': w_value_yinshi
                })
                explicit_faster_count += 1
            # 如果该节点在 edge_log.csv 中有连接关系，继续查找
            elif not found_earlier and not found_later:
                related_nodes = edge_log_df[
                    (edge_log_df['Target_Node'] == node_id)]  # 确保显式网络节点是起始点
                for _, edge_row in related_nodes.iterrows():
                    # 取得与该节点相关的另一节点的ID（隐式网络节点）
                    related_node_id = edge_row['Source_Node']

                    if related_node_id in xianshi_dict:
                        w_value_xianshi = xianshi_dict[related_node_id]
                        if w_value_yinshi > w_value_xianshi:  # 显式网络的时间更早
                            found_earlier = True
                            matched_nodes.append({
                                'xianshi_node_id': related_node_id,
                                'xianshi_w_value': w_value_xianshi,
                                'yinshi_node_id':  node_id,
                                'yinshi_w_value': w_value_yinshi
                            })
                            explicit_faster_count += 1
                            break  # 只处理第一次匹配，跳出相关节点循环
                        elif w_value_yinshi < w_value_xianshi:  # 隐式网络的时间更晚
                            found_later = True
                            break  # 找到一次晚于显式网络节点，跳出
                # 如果同时发现了比隐式网络节点时间早和比隐式网络节点时间晚的情况，跳过
                if found_earlier and found_later:
                    continue  # 跳过该节点，继续遍历下一个

# 将符合条件的节点存储到一个 DataFrame 中
matched_df = pd.DataFrame(matched_nodes)

# 计算比例
explicit_faster_ratio = explicit_faster_count / total_check_count if total_check_count > 0 else 0

# 保存符合条件的节点到 CSV 文件
matched_df.to_csv('matched_nodes_explicit_faster2.csv', index=False, encoding='utf-8')

# 打印结果
print(f"符合条件的节点已保存到 'matched_nodes_explicit_faster2.csv'！")
print(f"显式网络比隐式网络快的频次: {explicit_faster_count}")
print(f"显式网络比隐式网络快的比例: {explicit_faster_ratio:.2%}")
