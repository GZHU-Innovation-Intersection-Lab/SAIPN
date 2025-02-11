import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import numpy as np
from tqdm import tqdm
from community import best_partition  # pip install python-louvain
from networkx.algorithms.community import modularity
import csv
import random
from collections import deque
from networkx.readwrite.gexf import GEXFWriter
import re
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

def enhanced_main():
    """增强的主函数"""
    G = nx.DiGraph()

    # 读取数据并清理列名
    table = pd.read_csv("final_without_quotes.txt", encoding='utf-8')
    table.columns = table.columns.str.strip().str.replace(r'[\'"\s]', '', regex=True)
    
    edge_queue = deque(maxlen=100000)


    # 动态定位时间列名
    time_col_candidates = [col for col in table.columns if 'created_at' in col.lower()]
    if not time_col_candidates:
        raise KeyError("数据中无有效时间列")
    time_col = time_col_candidates[0]
    print(f"动态获取时间列名: {time_col}")  # 输出示例：raw_valuecreated_at
    
    # 时间处理流程
    table = table.dropna(subset=[time_col])
    table[time_col] = pd.to_datetime(table[time_col])
    table = table.set_index(time_col)
    
    # 必须使用 time_col 变量 ↓
    table = table.sort_values(time_col)  # 替换硬编码的 'raw_value.created_at'
    
    # 定义必需列匹配规则（允许列名包含点或下划线） ↓
    required_col_patterns = {
        'id_str': r'raw_value\.?id_?str',
        'user_id_str': r'raw_value\.?user_?id_?str',
        'in_reply_to_status_id_str': r'raw_value\.?in_?reply_?to_?status_?id_?str',
        'quoted_status_id_str': r'raw_value\.?quoted_?status_?id_?str'
    }
    
    # 动态匹配列名
    required_cols = {}
    for key, pattern in required_col_patterns.items():
        # 使用正则表达式模糊匹配 ↓
        matched_cols = [col for col in table.columns if re.search(pattern, col, re.IGNORECASE)]
        if not matched_cols:
            raise KeyError(f"无法匹配必需列: {pattern}")
        required_cols[key] = matched_cols[0]
    
    print("实际匹配的必需列:", required_cols)
    
    # 预处理检查（使用动态列名） ↓
    missing_cols = [key for key, col in required_cols.items() if col not in table.columns]
    if missing_cols:
        raise ValueError(f"缺失关键列: {missing_cols}")

    # 按天处理数据的主循环

    date_groups = table.groupby(pd.Grouper(freq='D'))
    
    with open('network_metrics.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'nodes', 'edges', 'modularity', 
                     'assortativity', 'intra_ratio', 'inter_ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for date, daily_data in date_groups:
            if daily_data.empty:
                continue
                
            print(f"\nProcessing {date.strftime('%Y-%m-%d')}...")
            
            # 删除过期边和节点
            expire_time = date - datetime.timedelta(days=30)
            while edge_queue and edge_queue[0]['timestamp'] < expire_time:
                expired_edge = edge_queue.popleft()
                if G.has_edge(expired_edge['source'], expired_edge['target']):
                    G.remove_edge(expired_edge['source'], expired_edge['target'])
                    
            # 添加当日节点和边
            current_nodes = set()
            for _, row in daily_data.iterrows():
                created_at = row.name
                iso_time = created_at.isoformat()
                try:
                    # 先获取 tweet_id ↓
                    tweet_id = str(row[required_cols['id_str']]).strip()
                    user_id = str(row[required_cols['user_id_str']]).strip()
                    reply_to = str(row[required_cols['in_reply_to_status_id_str']]).strip()
                    quoted = str(row[required_cols['quoted_status_id_str']]).strip()
                    
                    if not tweet_id or not user_id:
                        continue
                        
                    # 添加节点元数据
                    G.add_node(tweet_id, 
                            user_id=user_id,
                            created_at=created_at,
                            text=row.get('raw_valuefull_text', ''),
                            reply_to=reply_to,
                            quoted=quoted)
                    current_nodes.add(tweet_id)
                    
                    # 处理回复关系
                    reply_to = str(row['raw_value.in_reply_to_status_id_str']).strip()
                    if reply_to and reply_to != 'nan' and reply_to in G:
                        G.add_edge(tweet_id, reply_to)
                        edge_queue.append({
                            'source': tweet_id,
                            'target': reply_to,
                            'timestamp': created_at
                        })
                            
                    # 处理引用关系
                    quoted = str(row['raw_value.quoted_status_id_str']).strip()
                    if quoted and quoted != 'nan' and quoted in G:
                        G.add_edge(tweet_id, quoted)
                        edge_queue.append({
                            'source': tweet_id,
                            'target': quoted,
                            'timestamp': created_at
                        })
                            
                except Exception as e:
                    print(f"Error processing row {_}: {str(e)}")
                    continue
                    
            # 删除孤立节点（当日未被连接的节点）
            for node in list(G.nodes()):
                if node not in current_nodes and G.degree(node) == 0:
                    G.remove_node(node)
                    
            # 计算网络指标
            metrics = {
                'date': date.strftime('%Y-%m-%d'),
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges()
            }
            
            if metrics['edges'] > 0:
                try:
                    # 计算模块度
                    undirected_G = G.to_undirected()
                    partition = best_partition(undirected_G)
                    communities = [set(c) for c in set(partition.values())]
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
                               
            writer.writerow(metrics)
            
            # 可视化保存
            if G.number_of_nodes() > 1:
                plt.figure(figsize=(12, 6))
                pos = nx.spring_layout(G)
                node_colors = [G.nodes[n].get('community', 0) for n in G.nodes()]
                
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.tab20)
                nx.draw_networkx_edges(G, pos, alpha=0.2)
                plt.title(f"Network Structure on {date.strftime('%Y-%m-%d')}")
                plt.savefig(f"network_viz/{date.strftime('%Y%m%d')}.png")
                plt.close()
                
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
        enhanced_main()
    except Exception as e:
        print(f"致命错误: {str(e)}")
        import traceback
        traceback.print_exc()