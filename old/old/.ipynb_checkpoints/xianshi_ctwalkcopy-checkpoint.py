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
from community import best_partition
import random
import datetime
from collections import defaultdict

# 常量配置
CTWALKS_PARAMS = {
    'num_walks': 10,        # 每个时间窗口的随机游走次数
    'walk_length': 5,        # 每次游走的步长
    'sampling_rate': 0.3,    # 节点采样率
    'max_attempts': 3       # 无效节点重试次数
}

def validate_gexf_attributes(G):
    """确保所有属性符合GEXF规范"""
    valid_types = (str, int, float, bool)
    for node, data in G.nodes(data=True):
        for k, v in data.items():
            if not isinstance(v, valid_types):
                try:
                    # 尝试转换为字符串
                    data[k] = str(v)
                except:
                    del data[k]  # 删除无法转换的属性
    return G

def temporal_random_walk(G, start_node, walk_length, current_time):
    """时间感知的随机游走"""
    walk = [start_node]
    for _ in range(walk_length-1):
        # 获取时间有效的后续节点
        successors = []
        attempts = 0
        while not successors and attempts < CTWALKS_PARAMS['max_attempts']:
            try:
                successors = [
                    n for n in G.successors(walk[-1])
                    if 'created_at' in G.nodes[n] and 
                    G.nodes[n]['created_at'] >= current_time
                ]
                current_time = G.nodes[walk[-1]]['created_at']
            except KeyError:
                pass
            attempts += 1
        
        if not successors:
            break
            
        next_node = random.choice(successors)
        walk.append(next_node)
        
    return walk

def compute_ctwalks_metrics(G):
    """计算社区时间游走指标"""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0, 0.0
    
    # 社区检测
    undirected_graph = G.to_undirected()
    partition = best_partition(undirected_graph)
    nx.set_node_attributes(G, partition, 'community')
    
    # 节点采样
    all_nodes = list(G.nodes())
    sampled_nodes = random.sample(
        all_nodes, 
        k=int(len(all_nodes) * CTWALKS_PARAMS['sampling_rate'])
    ) if len(all_nodes) > 0 else []
    
    # 执行游走
    intra_count = 0
    inter_count = 0
    valid_walks = 0
    
    for start_node in sampled_nodes:
        if 'created_at' not in G.nodes[start_node]:
            continue
            
        walk = temporal_random_walk(
            G=G,
            start_node=start_node,
            walk_length=CTWALKS_PARAMS['walk_length'],
            current_time=G.nodes[start_node]['created_at']
        )
        
        if len(walk) < 2:
            continue
            
        valid_walks += 1
        for i in range(len(walk)-1):
            source = walk[i]
            target = walk[i+1]
            
            if 'community' not in G.nodes[source] or 'community' not in G.nodes[target]:
                continue
                
            if G.nodes[source]['community'] == G.nodes[target]['community']:
                intra_count += 1
            else:
                inter_count += 1
    
    total_steps = intra_count + inter_count
    if total_steps == 0:
        return 0.0, 0.0
    
    return intra_count/total_steps, inter_count/total_steps

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
    with open('network_metrics.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'nodes', 'edges', 'modularity', 
                     'assortativity', 'intra_ratio', 'inter_ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

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

    # 计算网络指标
                metrics = {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges()
                }
                
                if metrics['edges'] > 0:
                    try:
                        # 计算模块度
                        undirected_G = G.to_undirected()
                        partition = best_partition(undirected_G)
                        community_dict = defaultdict(set)
                        for node, comm_id in partition.items():
                            community_dict[comm_id].add(node)
                        communities = list(community_dict.values())
                        metrics['modularity'] = modularity(undirected_G, communities)
                        
                        # 计算同配性
                        metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
                        
                        # 计算CTWalks指标
                        intra, inter = compute_ctwalks_metrics(G)
                        metrics['intra_ratio'] = intra
                        metrics['inter_ratio'] = inter
                    except Exception as e:
                        print(f"指标计算错误: {str(e)}")
                        metrics.update({'modularity': 0, 'assortativity': 0,
                                    'intra_ratio': 0, 'inter_ratio': 0})
                else:
                    metrics.update({'modularity': 0, 'assortativity': 0,
                                'intra_ratio': 0, 'inter_ratio': 0})
                # 将指标写入CSV文件
                writer.writerow(metrics)

    # 最终清理
    G_clean = validate_gexf_attributes(G)
    nx.write_gexf(G, "final_network.gexf")
    print("处理完成。结果保存在network_metrics.csv和network_viz/目录")

if __name__ == '__main__':
    # 环境检查
    if not os.path.exists('network_viz'):
        os.makedirs('network_viz')
        
    if not os.path.isfile("final_without_quotes.txt"):
        raise FileNotFoundError("找不到输入文件final_without_quotes.txt")
    
    try:
        main2()
    except Exception as e:
        print(f"致命错误: {str(e)}")
        import traceback
        traceback.print_exc()

