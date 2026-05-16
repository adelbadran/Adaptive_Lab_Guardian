"""GNN spatial relationship helpers.

Torch Geometric is optional. When it is unavailable, the runtime pipeline uses
the lightweight `spatial_risk` fallback instead of failing at import time.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, global_mean_pool
except Exception:  # pragma: no cover - optional heavy dependency
    torch = None
    F = None
    GATConv = None
    global_mean_pool = None


SENSOR_NAMES = ["Temp", "Humidity", "Gas", "Light", "Motion"]


if torch is not None and GATConv is not None:

    class GATModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gat1 = GATConv(1, 16, heads=4)
            self.gat2 = GATConv(16 * 4, 8, heads=1)
            self.dropout = torch.nn.Dropout(0.3)

        def forward(self, x, edge_index, batch, return_attention=False):
            x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = self.dropout(x)
            x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
            x = self.dropout(x)
            x = global_mean_pool(x, batch)
            if return_attention:
                return x, attn1, attn2
            return x

else:
    GATModel = None


def spatial_risk(sample: np.ndarray) -> float:
    """Dependency-free spatial risk proxy from a filtered sensor vector."""
    x = np.asarray(sample, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.clip(np.std(x) + np.mean(x) * 0.25, 0.0, 1.0))


def get_attention_weights(model, sample, edge_index):
    if torch is None or model is None:
        return None, None

    model.eval()
    x = torch.tensor(sample, dtype=torch.float).view(len(sample), 1)
    batch = torch.zeros(len(sample), dtype=torch.long)
    with torch.no_grad():
        _, attn1, _ = model(x, edge_index, batch, return_attention=True)
    edge_idx, weights = attn1
    return edge_idx.cpu().numpy(), weights.cpu().numpy()


def draw_attention_graph(attention):
    """Return a matplotlib plot for dashboard debugging, if optional libs exist."""
    if attention is None:
        return None

    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return None

    edges = attention.get("edges", [])
    weights = attention.get("weights", [])
    graph = nx.DiGraph()

    for i, name in enumerate(SENSOR_NAMES):
        graph.add_node(i, label=name)

    for (src, dst), weight in zip(edges, weights):
        if src != dst:
            graph.add_edge(src, dst, weight=float(weight))

    pos = nx.spring_layout(graph, k=0.25, seed=42)
    _, ax = plt.subplots(figsize=(3, 3))
    nx.draw_networkx_nodes(graph, pos, node_size=700, ax=ax)
    edge_weights = [graph[u][v]["weight"] * 3 for u, v in graph.edges()]
    edge_colors = [graph[u][v]["weight"] for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, width=edge_weights, edge_color=edge_colors, edge_cmap=plt.cm.viridis, arrows=True)
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={(u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges()},
        font_size=5,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, labels={i: name for i, name in enumerate(SENSOR_NAMES)}, font_size=5)
    plt.title("GNN Attention Graph", fontsize=8)
    return plt
