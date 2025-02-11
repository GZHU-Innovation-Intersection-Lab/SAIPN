import pandas as pd
import os
import json
import networkx as nx
import datetime
import torch
from networkx.algorithms.community import greedy_modularity_communities, modularity
import sympy
from collections import deque
import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate
import csv 
def load_vectors(file_path):
    vector_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',', 1)  # 按逗号分割，第一部分是推文ID，第二部分是向量
            tweet_id = parts[0]  # 获取推文ID
            try:
                # 将向量部分按空格拆分并转换为浮动数值
                vector = np.array([float(x) for x in parts[1].split()])  # 向量部分按空格拆分
                vector_dict[tweet_id] = vector
            except ValueError as e:
                print(f"无法将以下值转换为浮动数值: {parts[1]}. 错误信息: {e}")
    return vector_dict

# 调用方法
vector_dict = load_vectors("output_vectors.txt")
print(f"已加载的推文ID数量: {len(vector_dict)}")
print(f"部分推文ID: {list(vector_dict.keys())[:10]}")  # 打印前10个ID


scores = 0.90
sentiment_df = pd.read_csv('final_with_sentiment.csv')

# 创建一个字典，方便通过 `id_str` 查找情感概率
sentiment_dict = {
    row['raw_value.id_str']: {
        'negative_probability': row['negative_probability'],
        'neutral_probability': row['neutral_probability'],
        'positive_probability': row['positive_probability']
    }
    for _, row in sentiment_df.iterrows()
}




def calculate_daw(time_diff_days):
    """根据时间差计算动态衰减指数DAW"""
    # 计算衰减因子，时间差转化为月单位
    time_diff_months = abs(time_diff_days) / 30

    # 使用数值积分替代符号积分
    result, error = integrate.quad(lambda x: np.exp(-x), 0, time_diff_months)
    
    # 计算DAW值，1减去积分结果
    daw_value = 1 - result
    
    # 保证DAW的下限为0.01
    return daw_value if daw_value >= 0.01 else 0.01



# 更新vol函数，整合DAW计算
def vol(parent_raws, emtions1, retweet_count, quote_count, favorite_count, reply_count,
        max_retweet_count, max_quote_count, max_favorite_count, max_reply_count, current_time):
    emtions = emtions1
    vol1 = 0
    if parent_raws:
        for raw in parent_raws:  # 0:相似度，1：相差时间，2：情感得分
            # 确保 raw[1] 是 pandas Timestamp 类型
            raw_time = pd.to_datetime(raw[1])  # 将父节点时间转换为 Timestamp 类型
            current_time = pd.to_datetime(current_time)  # 确保 current_time 是 Timestamp 类型
            
            # 计算时间差（天数）
            time_diff = (current_time - raw_time).days  # 获取天数差
            daw_value = calculate_daw(time_diff)  # 计算DAW
            vol1 += daw_value * raw[0] * raw[2]  # 根据DAW、相似度、情感得分计算传播强度
        vol1 /= len(parent_raws)
    # 父节点的平均情感 + 子节点情感 * 子节点的相关指标
    print('emotion:', emtions, 'volumn:', vol1)
    emtions += vol1
    if emtions < -1:
        emtions = -1
    elif emtions > 1:
        emtions = 1
    return emtions

def emtions(tweet_id):
    # 直接通过 tweet_id 获取情感概率
    sentiment_probs = sentiment_dict.get(tweet_id)
    if sentiment_probs:
        negative_prob = sentiment_probs['negative_probability']
        neutral_prob = sentiment_probs['neutral_probability']
        positive_prob = sentiment_probs['positive_probability']

        # 计算情感分数
        if positive_prob > neutral_prob and positive_prob > negative_prob:
            sentiment_score = positive_prob
        elif negative_prob > neutral_prob and negative_prob > positive_prob:
            sentiment_score = -negative_prob
        elif neutral_prob > positive_prob and neutral_prob > negative_prob:
            if positive_prob > negative_prob:
                sentiment_score = 1 - neutral_prob
            else:
                sentiment_score = -(1 - neutral_prob)
    else:
        sentiment_score = 0  # 如果找不到情感概率，返回默认值（例如0）

    return sentiment_score

def bert_va(tweet_id, emtions1):
    tweet_id = str(tweet_id)  # 确保tweet_id是字符串类型
    if tweet_id not in vector_dict:
        raise ValueError(f"向量文件中找不到推文ID {tweet_id}")
    
    tweet_vector = vector_dict[tweet_id]
    embedding_with_emotion = np.concatenate((tweet_vector, np.array([emtions1])), axis=0)
    
    return embedding_with_emotion


def cosine(x, y):
    # 确保x和y是tensor类型
    x = torch.tensor(x)
    y = torch.tensor(y)
    
    # 计算余弦相似度
    return torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))

# Assuming these functions are already defined elsewhere in your code
# emtions(), bert_va(), cosine(), calculate_daw(), vol(), etc.

import csv
from collections import deque

def main2():
    G = nx.DiGraph()

    # 读取数据并进行初步处理
    table = pd.read_csv("final.txt")
    table = table.iloc[:, 1:]  # 删除第一列，如果是索引列
    table['raw_value.created_at'] = pd.to_datetime(table['raw_value.created_at'])

    # 确保数据按创建时间排序
    table = table.sort_values(by='raw_value.created_at')

    table.index = table['raw_value.created_at']  # 将时间列设置为索引

    # 初始化数据结构
    users = {i: {} for i in set(table['raw_value.user_id_str'].values.tolist())}
    makelist = {}
    emotionlist = {}
    date_diff = 30  # 删除节点的时间差上限为30天

    # 获取最大转发、引用、点赞和评论数量
    max_retweet_count, max_quote_count, max_favorite_count, max_reply_count = (
        table['raw_value.retweet_count'].max(),
        table['raw_value.quote_count'].max(),
        table['raw_value.favorite_count'].max(),
        table['raw_value.reply_count'].max()
    )

    tweet_to_user = {}

    # 使用双端队列缓存边的时间戳和对应的边
    edge_timestamps = deque()  # 用于存储边的时间戳和对应的边

    # 用于缓存时间段的计算结果
    data_buffer = []  # 存储每个时间点的指标数据

    # 打开 edge_log.csv 文件，用于记录连边信息
    with open("edge_log.csv", "a+", newline='', encoding="utf-8") as edge_file:
        edge_writer = csv.writer(edge_file)
        # 如果文件为空，写入表头
        if edge_file.tell() == 0:
            edge_writer.writerow(['Source_Node', 'Target_Node', 'Timestamp'])

        # 按天分组数据进行处理
        for w, w_t in table.resample('D'):  # 每天处理一次数据
            if len(w_t) == 0:
                continue  # 如果这一天没有数据，跳过该天的处理

            current_time = w  # 使用当前时间作为当前时间
            new_edges_added = False  # 标记这一天是否有新增边或节点

            # 处理每一条推文数据
            for index, row in w_t.iterrows():
                strs = row['raw_value.full_text']
                strs2 = row['Tags']
                tweet_id = row['raw_value.id_str']  # 获取 tweet_id

                # 获取情感分数
                emtions1 = emtions(tweet_id)
                bert_va1 = bert_va(tweet_id, emtions1)
                this_data_datetime = row['raw_value.created_at']

                G.add_node(row['raw_value.id_str'])
                tweet_to_user[row['raw_value.id_str']] = row['raw_value.user_id_str']
                parent = []  # 存储相关的父节点

                # 遍历当前的 makelist，检查已有的边是否超过30天
                for make_i in makelist.keys():
                    diff = (this_data_datetime - makelist[make_i]['created_at']).days
                    if diff < date_diff:  # 如果时间差小于30天，继续检查
                        daw_value = calculate_daw(diff)
                        score = float(cosine(bert_va1, makelist[make_i]['pooled']) * daw_value)

                        if 1 > score > scores:  # 相似度阈值
                            # 添加带有时间戳的边
                            G.add_edge(makelist[make_i]['id'], row['raw_value.id_str'], weight=round(score, 2), timestamp=this_data_datetime)
                            
                            # 记录边的ID和时间戳到文件
                            edge_writer.writerow([makelist[make_i]['id'], row['raw_value.id_str'], this_data_datetime])
                            new_edges_added = True

                # 更新情感值
                emtions1 = vol(parent, emtions1, row['raw_value.retweet_count'], row['raw_value.quote_count'],
                               row['raw_value.favorite_count'], row['raw_value.reply_count'],
                               max_retweet_count, max_quote_count, max_favorite_count, max_reply_count, current_time)

                if w not in users[row['raw_value.user_id_str']].keys():
                    users[row['raw_value.user_id_str']][w] = [emtions1]
                else:
                    users[row['raw_value.user_id_str']][w].append(emtions1)

                # 更新节点信息
                makelist[row['raw_value.id_str']] = {
                    'userid': row['raw_value.user_id_str'],
                    'id': row['raw_value.id_str'],
                    'text': row['raw_value.full_text'],
                    'cleantext': strs,
                    'emotion': emtions1,
                    'pooled': bert_va1,
                    'created_at': row['raw_value.created_at']
                }

            # 即使没有新节点或边，也要删除超过30天的节点和它们的边
            to_remove_nodes = set()  # 用于存储需要删除的节点

            # 检查每个节点是否存在超过30天
            for node in list(G.nodes):  # 使用 list() 避免在遍历中修改节点集合
                node_creation_time = makelist[node]['created_at']
                node_age = (current_time - node_creation_time).days
                if node_age > date_diff:  # 如果节点存在超过30天
                    to_remove_nodes.add(node)

            # 删除超过30天的节点及其所有边
            for node in tqdm(to_remove_nodes, desc="删除过期节点", unit="节点"):
                if G.has_node(node):
                    # 删除与该节点相关的所有边
                    G.remove_node(node)
                    print(f"已删除节点及其所有边：{node}  (时间差超过 {date_diff} 天)")

            # 删除节点和边后，及时更新数据结构
            for node in to_remove_nodes:
                if node in makelist:
                    del makelist[node]
                if node in tweet_to_user:
                    del tweet_to_user[node]

            # 计算网络指标并缓存到内存
            if G.number_of_edges() > 0:
                d = dict(nx.degree(G))
                d_avg = sum(d.values()) / len(G.nodes)

                x = list(range(max(d.values()) + 1))
                d_list = nx.degree_histogram(G)
                y = np.array(d_list) / len(G)

                # 检测社区并计算模块化程度
                communities = greedy_modularity_communities(G.to_undirected())
                modularity_score = modularity(G.to_undirected(), communities, weight=None)

                q1 = G.number_of_nodes()  # 节点数
                q2 = G.number_of_edges()  # 边数
                q3 = nx.pagerank(G,alpha=0.85)
                q6 = nx.degree_assortativity_coefficient(G)  # 度同配系数

                # 将计算结果缓存到 data_buffer
                data_buffer.append([w.strftime('%Y-%m-%d'), q1, q2, q3, q6, modularity_score])

            # 批量写入文件
            if len(data_buffer) >= 10:  # 如果缓存的记录达到10个（可以根据需要调整）
                with open("index.csv", 'a+', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:  # 如果文件为空，先写入表头
                        writer.writerow(['Time', 'Num_nodes', 'Num_edges', 'PageRank', 'Degree_assortativity', 'Modularity'])
                    writer.writerows(data_buffer)  # 批量写入
                data_buffer.clear()  # 清空缓存

                # 可选：输出日志
                print(f"已批量写入 {len(data_buffer)} 条数据到 index.csv")

            timestamp_str = w.strftime("%Y-%m-%d")
            adjacency_matrix = nx.adjacency_matrix(G).todense()
            pd.DataFrame(adjacency_matrix, index=G.nodes, columns=G.nodes).to_csv(f'G-{timestamp_str}.csv')

            tweet_network_filename = f'tweet_network-{timestamp_str}.edgelist'
            nx.write_edgelist(G, tweet_network_filename)

            with open("final_emotions.csv", 'a+') as f:
                f.write(f"{timestamp_str},{emtions1}\n")

    # 保存映射关系和其他信息
    with open("emotionall.csv", "w+", encoding='utf-8') as f:
        f.write(str(makelist))

    # 保存tweet_to_user映射
    with open("tweet_to_user_mapping.json", "w") as f:
        json.dump(tweet_to_user, f)

    return users




def main3():
    # 从文件名中提取时间段
    time_periods = []
    for file in os.listdir('.'):
        if file.startswith("tweet_network-") and file.endswith(".edgelist"):
            time_periods.append(file.replace("tweet_network-", "").replace(".edgelist", ""))
    
    # 加载全局推文到用户的映射
    with open("tweet_to_user_mapping.json", "r") as f:
        tweet_to_user = json.load(f)
    
    for period in sorted(time_periods):  # 按时间顺序处理
        tweet_network_file = f"tweet_network-{period}.edgelist"
        
        # 读取推文网络
        G_tweet = nx.read_edgelist(tweet_network_file, data=(("weight", float),))

        # 构造用户网络
        G_user = nx.DiGraph()
        for tweet1, tweet2, weight in G_tweet.edges(data="weight"):
            if tweet1 in tweet_to_user and tweet2 in tweet_to_user:
                user1, user2 = tweet_to_user[tweet1], tweet_to_user[tweet2]

                if user1 != user2:  # 排除同用户间的连接
                    if G_user.has_edge(user1, user2):
                        G_user[user1][user2]["weight"] += weight
                    else:
                        G_user.add_edge(user1, user2, weight=weight)

        # 保存用户网络的邻接矩阵
        adjacency_matrix = nx.adjacency_matrix(G_user).todense()
        pd.DataFrame(adjacency_matrix, index=G_user.nodes, columns=G_user.nodes).to_csv(f"G-用户网络-{period}.csv")

        # 绘制用户网络图
        pos = nx.spring_layout(G_user)
        plot_networkx(G_user, pos, filename=f"user_network-{period}.png")

        # 计算 PageRank 并保存
        user_pagerank = nx.pagerank(G_user, alpha=0.85)
        user_pagerank_sorted = sorted(user_pagerank.items(), key=lambda d: d[1], reverse=True)

        with open(f"index_user-{period}.csv", "w") as f:
            for user, score in user_pagerank_sorted:
                f.write(f"{user},{score}\n")







# 主函数执行
if __name__ == '__main__':
    print(datetime.datetime.now())
    main2()
