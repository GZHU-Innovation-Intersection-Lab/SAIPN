import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import datetime
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
import numpy as np
from sympy.abc import x
from networkx.algorithms.community import greedy_modularity_communities, modularity
from tqdm import tqdm
from collections import deque
import csv 
def is_valid_id(tweet_id):
    # 检查推文ID是否是有效的（例如不为空，且是数字，或者是合理的ID范围）
    if pd.isna(tweet_id) or tweet_id == "" or not isinstance(tweet_id, (int, float)):
        return False
    return True
def main2():
    # 创建有向图
    G = nx.DiGraph()

    # 读取数据并预处理
    table = pd.read_csv("final_without_quotes.txt", encoding='utf-8')
    table = table.iloc[:, 1:]  # 移除第一列（索引列）

    # 处理时间字段，删除无效或格式错误的记录
    to_drop = []  # 用于存储需要删除的行索引
    for i in range(len(table)):
        try:
            table.loc[i, 'raw_value.created_at'] = pd.to_datetime(table.loc[i, 'raw_value.created_at'])
        except:
            to_drop.append(i)
        if isinstance(table.loc[i, 'raw_value.created_at'], datetime.datetime):
            pass
        else:
            to_drop.append(i)

        if table.loc[i, 'raw_value.created_at'].tzinfo is not None:
            to_drop.append(i)

    # 删除无效的行
    table.drop(to_drop, inplace=True)
    
    table['raw_value.created_at'].sort_values(ascending=True)
    table.to_csv("final_without_quotes.txt", encoding='utf-8')
    table.index = table['raw_value.created_at']

    # 定义数据存储结构
    users = {i: {} for i in set(table['raw_value.user_id_str'].values.tolist())}
    makelist = {}
    date_diff = 30  # 时间差阈值（30天）
    output_dir = "degree"
    edge_timestamps = deque()  # 双端队列缓存边的时间戳和对应的边
    edge_lifetime = 30  # 边的生命周期 (天)
    
    # 用于批量存储index.csv中的数据
    index_buffer = []  # 用于存储一段时间内的写入数据

    # 按天处理数据
    for w, w_t in table.resample('D'):  # 按天分组
        if len(w_t) > 0:  # 如果本天有数据
            for index, row in w_t.iterrows():
                tweet_id = row['raw_value.id_str']  # 获取推文ID
                # 判断推文是否存在交互关系
                if row['raw_value.in_reply_to_status_id_str'] or row['raw_value.quoted_status_id_str']:
                    this_data_datetime = row['raw_value.created_at']
                    
                    # 检查用户ID是否为空
                    user_id = row['raw_value.user_id_str']
                    if pd.isna(user_id) or user_id == "":
                        continue  # 跳过没有有效用户ID的记录

                    # 添加节点到图中
                    G.add_node(tweet_id)  # 添加节点到图中

                    # 遍历缓存中的推文，寻找交互关系
                    for make_i in makelist.keys():
                        diff = (this_data_datetime - makelist[make_i]['created_at']).days
                        if diff < date_diff:  # 如果时间差在范围内
                            # 回复关系
                            if is_valid_id(row['raw_value.in_reply_to_status_id_str']):
                                edge_timestamps.append((makelist[make_i]['id'], row['raw_value.in_reply_to_status_id_str'], this_data_datetime))
                                G.add_edge(makelist[make_i]['id'], row['raw_value.in_reply_to_status_id_str'], weight=1, timestamp=this_data_datetime)

                            if is_valid_id(row['raw_value.quoted_status_id_str']):
                                edge_timestamps.append((makelist[make_i]['id'], row['raw_value.quoted_status_id_str'], this_data_datetime))
                                G.add_edge(makelist[make_i]['id'], row['raw_value.quoted_status_id_str'], weight=1, timestamp=this_data_datetime)

            if row['raw_value.user_id_str'] not in makelist:
                makelist[row['raw_value.user_id_str']] = {
                    'userid': row['raw_value.user_id_str'],
                    'id': tweet_id,
                    'text': row['raw_value.full_text'],
                    'reply': row['raw_value.in_reply_to_status_id_str'],
                    'quoted': row['raw_value.quoted_status_id_str'],
                    'cleantext': row['raw_value.full_text'],
                    'created_at': row['raw_value.created_at']
                }

            # 删除超过30天的边和节点
            to_remove_edges = []
            to_remove_nodes = set()  # 用于存储需要删除的节点

            # 从双端队列中删除过期的边
            while edge_timestamps and (w - edge_timestamps[0][2]).days > edge_lifetime:
                edge_start, edge_end, timestamp = edge_timestamps.popleft()
                if G.has_edge(edge_start, edge_end):
                    G.remove_edge(edge_start, edge_end)
                    to_remove_nodes.add(edge_start)
                    to_remove_nodes.add(edge_end)
                    print(f"已删除边：{(edge_start, edge_end)}  (时间差超过 {date_diff} 天)")

            # 删除节点
            for node in tqdm(to_remove_nodes, desc="删除过期节点", unit="节点"):
                if G.has_node(node):
                    G.remove_node(node)
                    print(f"已删除节点：{node}  (时间差超过 {date_diff} 天)")



            # 计算度分布并保存图片
            if G.number_of_nodes() > 0:  # 确保图中有节点
                d = dict(nx.degree(G))  # 计算每个节点的度
                d_avg = sum(d.values()) / len(G.nodes)  # 计算平均度

                # 获取所有的度值及其对应的概率
                x = list(range(max(d.values()) + 1))
                d_list = nx.degree_histogram(G)  # 获取每个度值出现的次数
                y = np.array(d_list) / len(G)  # 计算每个度值的出现概率

                # 绘制度分布图
                plt.figure(figsize=(8, 6))
                plt.plot(x, y, 'o-', label=f'Avg Degree: {d_avg:.2f}')
                plt.xlabel('Degree')
                plt.ylabel('Probability')
                plt.title(f'Degree Distribution for Day {w.strftime("%Y-%m-%d")}')
                plt.legend()
                plt.grid()

                # 保存图片到文件夹
                output_path = os.path.join(output_dir, f"degree_{w.strftime('%Y-%m-%d')}.png")
                plt.savefig(output_path)

            # After processing each day, output network metrics
            if G.number_of_edges() > 0:
                d = dict(nx.degree(G))
                d = {node: (degree if not np.isnan(degree) else 0) for node, degree in d.items()}
                d_avg = sum(d.values()) / len(G.nodes)

                x = list(range(max(d.values()) + 1))
                d_list = nx.degree_histogram(G)
                y = np.array(d_list) / len(G)

                # 检测社区并计算模块化程度
                # 转换为无向图
                undirected_G = G.to_undirected()

                # 初始化模块化得分
                modularity_score = 0  # 初始化确保modularity_score有值

                # 检查图是否为空或没有边
                if undirected_G.number_of_edges() == 0:
                    print("Graph has no edges. Setting modularity to 0.")
                    modularity_score = 0
                else:
                    # 进行社区检测
                    try:
                        communities = list(greedy_modularity_communities(undirected_G))
                        if communities:
                            modularity_score = modularity(undirected_G, communities)
                            print(f"Modularity score: {modularity_score}")
                        else:
                            print("No communities detected. Setting modularity to 0.")
                            modularity_score = 0
                    except Exception as e:
                        print(f"Error during community detection: {e}")
                        modularity_score = 0
            else:
                print("Graph has no edges. Setting modularity to 0.")
                modularity_score = 0

            q1 = G.number_of_nodes()
            q2 = G.number_of_edges()
            q3 = nx.pagerank(G,alpha=0.85)
            q6 = nx.degree_assortativity_coefficient(G)
            q4 = modularity_score
			
            # Store the current index information
            index_buffer.append([str(w), str(q1), str(q2), str(q3), str(q6),str(q4)])

            # Write to CSV when buffer reaches a certain size (e.g., every 100 days)
            if len(index_buffer) >= 100:
                with open('index.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:  # Check if file is empty to write headers
                        writer.writerow(['Time', 'Num_nodes', 'Num_edges', 'Dessity', 'Modularity'])
                    writer.writerows(index_buffer)
                index_buffer.clear()  # Clear the buffer after writing

    # Write any remaining data to the CSV file
    if index_buffer:
        with open('index.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Check if file is empty to write headers
                writer.writerow(['Time', 'Num_nodes', 'Num_edges', 'Avg_clustering', 'Modularity'])
            writer.writerows(index_buffer)

            # Save adjacency matrix for the current week
            timestamp_str = w.strftime("%Y-%m-%d")
            adjacency_matrix = nx.adjacency_matrix(G).todense()
            pd.DataFrame(adjacency_matrix, index=G.nodes, columns=G.nodes).to_csv(f'G-{timestamp_str}.csv')

    # Save final data to a text file
    with open("normaldata2.txt", "w+", encoding='utf-8') as f:
        f.write(str(makelist))
    # Save the final adjacency matrix
    d = nx.adjacency_matrix(G).todense()
    pd.DataFrame(d, index=G.nodes, columns=G.nodes).to_csv('G.csv')

    return makelist


if __name__ == '__main__':
    print(datetime.datetime.now())
    main2()

