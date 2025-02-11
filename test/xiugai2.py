import pandas as pd

# 读取 top5_xianshi.csv 和 top5_yinshi.csv
df_xianshi = pd.read_csv('top10_percent_xianshi.csv')
df_yinshi = pd.read_csv('top10_percent_yinshi.csv')

# 读取 edge_log.csv 文件，存储每一条边的连接信息
edge_log_df = pd.read_csv('edge_log.csv')

# 用于存储符合条件的节点
matched_nodes = []

# 计数器：统计显式网络比隐式网络快的频次
explicit_faster_count = 0
total_check_count = 0  # 用于计算比例

# 计算节点总数
total_xianshi_nodes = len(df_xianshi)
total_yinshi_nodes = len(df_yinshi)

# 遍历 top5_xianshi.csv，检查每个显式网络节点
for index_x, row_xianshi in df_xianshi.iterrows():
    w_value_xianshi = row_xianshi['w_value']
    top5_nodes_xianshi = eval(row_xianshi['top10_percent_nodes'])  # 解析节点列表
    print(f"  Checking explicit network node {index_x + 1}/{total_xianshi_nodes}...")
    
    # 遍历显式网络节点
    for node_id_x, _ in top5_nodes_xianshi:
        total_check_count += 1  # 每检查一个显式网络节点，计数增加

        # 用来记录是否找到显式网络节点的时间既比隐式网络早又比隐式网络晚
        found_earlier = False
        found_later = False

        # 遍历 top5_yinshi.csv，检查每个隐式网络节点
        for index_y, row_yinshi in df_yinshi.iterrows():
            w_value_yinshi = row_yinshi['w_value']
            top5_nodes_yinshi = eval(row_yinshi['top10_percent_nodes'])  # 解析节点列表
            print(f"Processing implicit network node {index_y + 1}/{total_yinshi_nodes}...")

            # 遍历隐式网络节点进行时间比较
            for node_id_y, _ in top5_nodes_yinshi:
                if node_id_x == node_id_y:  # 如果显式网络节点和隐式网络节点相同
                    if w_value_yinshi < w_value_xianshi:  # 隐式网络节点比显式网络节点时间早
                        found_later = True
                    elif w_value_yinshi > w_value_xianshi:  # 隐式网络节点比显式网络节点时间晚
                        found_earlier = True

                    # 如果没有同时发现比显式网络节点早和比显式网络节点晚的情况，则继续检查
                    if not (found_earlier and found_later):
                        if found_earlier:
                            matched_nodes.append({
                                'xianshi_node_id': node_id_x,
                                'xianshi_w_value': w_value_xianshi,
                                'yinshi_node_id': node_id_y,
                                'yinshi_w_value': w_value_yinshi
                            })
                            explicit_faster_count += 1
                        elif not found_earlier and not found_later:
                            # 如果该节点在 edge_log.csv 中有连接关系，继续查找
                            related_nodes = edge_log_df[
                                (edge_log_df['Target_Node'] == node_id_x)]  # 确保显式网络节点是目标节点
                            for _, edge_row in related_nodes.iterrows():
                                # 取得与该节点相关的另一节点的ID（隐式网络节点）
                                related_node_id = edge_row['Source_Node']
                                if related_node_id == node_id_y:
                                    if w_value_yinshi > w_value_xianshi:  # 显式网络的时间更早
                                        matched_nodes.append({
                                            'xianshi_node_id': related_node_id,
                                            'xianshi_w_value': w_value_xianshi,
                                            'yinshi_node_id': node_id_y,
                                            'yinshi_w_value': w_value_yinshi
                                        })
                                        explicit_faster_count += 1
                                        break  # 只处理第一次匹配，跳出相关节点循环
                                    elif w_value_yinshi < w_value_xianshi:  # 隐式网络的时间更晚
                                        found_later = True
                                        break  # 找到一次晚于显式网络节点，跳出
                    # 如果同时发现了比显式网络节点时间早和比显式网络节点时间晚的情况，跳过
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

