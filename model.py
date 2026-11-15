"""Edge-aware GraphSAGE (pure PyTorch, no torch-geometric needed)."""
import torch, torch.nn as nn, torch.nn.functional as F

class EdgeSAGELayer(nn.Module):
    def __init__(self, node_in, edge_in, out):
        super().__init__()
        self.msg = nn.Linear(node_in + edge_in, out)
        self.upd = nn.Linear(node_in + out, out)

    def forward(self, h, ei, ea):
        src, dst = torch.cat([ei[0], ei[1]]), torch.cat([ei[1], ei[0]])   # both directions
        ea2 = torch.cat([ea, ea])
        m = F.relu(self.msg(torch.cat([h[src], ea2], 1)))                 # message = neighbour + flow features
        agg = torch.zeros(h.size(0), m.size(1), device=h.device).index_add_(0, dst, m)
        deg = torch.zeros(h.size(0), device=h.device).index_add_(0, dst, torch.ones(dst.size(0), device=h.device))
        agg = agg / deg.clamp(min=1).unsqueeze(1)                          # mean aggregation
        return F.relu(self.upd(torch.cat([h, agg], 1)))

class BotnetGNN(nn.Module):
    def __init__(self, feat_dim, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        dims = [feat_dim] + [hidden] * layers
        self.layers = nn.ModuleList(EdgeSAGELayer(dims[i], feat_dim, dims[i + 1]) for i in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden + feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x, ei, ea):
        h = x
        for layer in self.layers:
            h = self.drop(layer(h, ei, ea))
        return self.head(torch.cat([h[ei[0]], h[ei[1]], ea], 1))          # classify every flow (edge)
