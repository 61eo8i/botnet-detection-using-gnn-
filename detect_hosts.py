"""Scores rows of a per-host feature file (CTU/NCC format) with the trained HostGNN."""
import numpy as np, torch
from train_hosts import HostGNN, raw_features, make_graphs

class HostDetector:
    def __init__(self, path="models/host_gnn.pt"):
        ck = torch.load(path, weights_only=False, map_location="cpu"); self.ck = ck
        self.model = HostGNN(len(ck["cols"]), ck["hidden"]); self.model.load_state_dict(ck["state"]); self.model.eval()

    def score(self, df, window=4000):
        ck = self.ck
        X = ((raw_features(df, ck["cols"]) - ck["mean"]) / ck["std"]).astype(np.float32)
        p = np.zeros(len(df), dtype=np.float32)
        with torch.no_grad():
            for g in make_graphs(df, X, None, window, ck["k"]):
                p[g["idx"]] = torch.softmax(self.model(g["x"], g["ei"]), 1)[:, 1].numpy()
        return p
