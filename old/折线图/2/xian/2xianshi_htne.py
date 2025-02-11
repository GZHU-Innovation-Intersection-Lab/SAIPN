import htne  
import networkx as nx  
from datetime import datetime  

def build_htne_graph(edges_with_time):  
    """  
    将边列表转换为 HTNE 图对象  
    :param edges_with_time: [(u, v, timestamp)]  
    """  
    G = htne.Graph()  
    nodes = set()  
    for u, v, t in edges_with_time:  
        nodes.add(u)  
        nodes.add(v)  
        G.add_edge(u, v, int(t.timestamp()))  # HTNE 要求时间戳为整数  
    return G  

def run_htne_embedding(G, dim=128):  
    """  
    生成 HTNE 嵌入  
    """  
    model = htne.HTNE(G,  
                     dim=dim,  
                     walk_length=10,  
                     num_walks=20,  
                     workers=4)  
    embeddings = model.train()  
    return embeddings  

# 示例调用  
edges = [("userA", "userB", datetime(2023,5,1)),  
         ("userC", "userD", datetime(2023,5,2))]  
htne_graph = build_htne_graph(edges)  
embeddings = run_htne_embedding(htne_graph)  
print(embeddings["userA"][:5])  # 输出前5维嵌入