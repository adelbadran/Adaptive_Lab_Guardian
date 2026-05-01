import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt
from ai.preprocessing import preprocess_data
from torch_geometric.data import Data

class GATModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gat1 = GATConv(1, 16, heads=4)
        self.gat2 = GATConv(16 * 4, 8, heads=1)

        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index, batch,return_attention=False):
        # GAT layer 1
        x, attn1  = self.gat1(x, edge_index,return_attention_weights=True)
        x = F.elu(x)
        x = self.dropout(x)
        
        # GAT layer 2
        x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        if return_attention:
            return x, attn1, attn2

        return x
    
def get_attention_weights(model, sample, edge_index):
    model.eval()

    x = torch.tensor(sample, dtype=torch.float).view(5, 1)
    batch = torch.zeros(5, dtype=torch.long)

    with torch.no_grad():
        _, attn1, attn2 = model(x, edge_index, batch, return_attention=True)

    edge_idx, weights = attn1  # use first layer

    return edge_idx.numpy(), weights.numpy()


SENSOR_NAMES = ["Temp", "Humidity", "Gas", "Light", "Motion"]

def draw_attention_graph(attention):
    if attention is None:
        return None

    edges = attention["edges"]
    weights = attention["weights"]

    G = nx.DiGraph()

    # Add nodes
    for i, name in enumerate(SENSOR_NAMES):
        G.add_node(i, label=name)

    # Add edges with weights
    for (src, dst), w in zip(edges, weights):
        G.add_edge(src, dst, weight=w)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(6, 6))

    nx.draw_networkx_nodes(G, pos, node_size=2000)
    nx.draw_networkx_labels(G, pos)

    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, arrows=True)

    plt.title("GNN Attention Graph")

    return plt   