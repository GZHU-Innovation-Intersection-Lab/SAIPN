import pandas as pd

# 读取 top5_yinshi.csv 和 top5_xianshi.csv
df_yinshi = pd.read_csv('top10_percent_yinshi.csv')  # 隐式网络节点数据
df_xianshi = pd.read_csv('top10_percent_xianshi.csv')  # 显式网络节点数据

# 用于存储隐式网络节点及其对应的时间 (w_value)
yinshi_dict = {}

# 遍历 top5_yinshi.csv，将节点 ID 和对应的 w_value 存储到字典中
for _, row in df_yinshi.iterrows():
    w_value_yinshi = row['w_value']  # 获取隐式网络时间值
    yinshi_nodes = eval(row['top10_percent_nodes'])  # 解析节点列表字符串为实际的列表对象
    for yinshi_node_id, _ in yinshi_nodes:  # 遍历每个节点及其附加信息
        if yinshi_node_id not in yinshi_dict:  # 如果节点尚未存储，添加到字典
            yinshi_dict[yinshi_node_id] = w_value_yinshi

# 读取 edge_log.csv 文件（包含显式和隐式网络的连接关系）
edge_log_df = pd.read_csv('edge_log.csv')

# 用于存储符合条件的节点
matched_nodes = []
# 隐式网络比显式网络快的计数
implicit_faster_count = 0
# 总检查的节点数
total_check_count = 0

# 遍历 top5_xianshi.csv，检查显式网络中的节点是否满足条件
for _, row in df_xianshi.iterrows():
    w_value_xianshi = row['w_value']  # 获取显式网络时间值
    xianshi_nodes = eval(row['top10_percent_nodes'])  # 解析节点列表字符串为实际的列表对象

    # 遍历显式网络的每个节点
    for xianshi_node_id, _ in xianshi_nodes:
        total_check_count += 1  # 增加检查的节点计数

        # 标记当前显式节点是否比隐式网络节点时间更早或更晚
        found_earlier = False
        found_later = False

        # 检查显式网络节点是否也存在于隐式网络中
        if xianshi_node_id in yinshi_dict:
            w_value_yinshi = yinshi_dict[xianshi_node_id]  # 获取隐式网络对应节点的时间值
            if w_value_yinshi < w_value_xianshi:  # 如果隐式网络的时间更早
                found_earlier = True
            elif w_value_yinshi > w_value_xianshi:  # 如果隐式网络的时间更晚
                found_later = True

        # 如果没有同时发现时间更早和更晚的情况
        if not (found_earlier and found_later):
            # 如果隐式网络比显式网络快，记录该节点
            if found_earlier:
                matched_nodes.append({
                    'yinshi_node_id': xianshi_node_id,
                    'yinshi_w_value': w_value_yinshi,
                    'xianshi_node_id': xianshi_node_id,
                    'xianshi_w_value': w_value_xianshi
                })
                implicit_faster_count += 1
            # 如果显式节点没有直接在隐式网络中找到，检查其关联节点
            elif not found_earlier and not found_later:
                # 查找 edge_log.csv 中与当前显式节点相关的所有隐式节点
                related_nodes = edge_log_df[
                    (edge_log_df['Target_Node'] == xianshi_node_id)]
                for _, edge_row in related_nodes.iterrows():
                    related_node_id = edge_row['Source_Node']  # 获取关联的隐式网络节点 ID
                    if related_node_id in yinshi_dict:
                        w_value_yinshi = yinshi_dict[related_node_id]  # 获取隐式网络的时间值
                        if w_value_yinshi < w_value_xianshi:  # 如果关联的隐式网络时间更早
                            found_earlier = True
                            matched_nodes.append({
                                'yinshi_node_id': related_node_id,
                                'yinshi_w_value': w_value_yinshi,
                                'xianshi_node_id': xianshi_node_id,
                                'xianshi_w_value': w_value_xianshi
                            })
                            implicit_faster_count += 1
                            break  # 找到一个符合条件的关联节点后跳出
                        elif w_value_yinshi > w_value_xianshi:  # 如果隐式网络时间更晚
                            found_later = True
                            break
                # 如果同时发现更早和更晚的情况，跳过当前节点
                if found_earlier and found_later:
                    continue

# 将符合条件的节点保存为 DataFrame
matched_df = pd.DataFrame(matched_nodes)
# 计算隐式网络比显式网络快的比例
implicit_faster_ratio = implicit_faster_count / total_check_count if total_check_count > 0 else 0

# 将结果保存为 CSV 文件
matched_df.to_csv('matched_nodes_implicit_faster2.csv', index=False, encoding='utf-8')

# 打印结果
print(f"符合条件的节点已保存到 'matched_nodes_implicit_faster2.csv'！")
print(f"隐式网络比显式网络快的频次: {implicit_faster_count}")
print(f"隐式网络比显式网络快的比例: {implicit_faster_ratio:.2%}")
