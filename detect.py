import numpy as np, pandas as pd, torch
from data import prepare, build_graphs, find, SRC, DST
from model import BotnetGNN

class Detector:
    def __init__(self, path="models/botnet_gnn.pt"):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        self.scaler = ck["scaler"]
        self.model = BotnetGNN(ck["feat_dim"], ck["hidden"]); self.model.load_state_dict(ck["state"]); self.model.eval()

    def analyze(self, df, threshold=0.5, min_flows=5, min_ratio=0.5, window=5000):
        df, X, _, _ = prepare(df, self.scaler)
        prob = np.zeros(len(df), dtype=np.float32)
        with torch.no_grad():
            for g in build_graphs(df, X, None, window):
                prob[g["idx"]] = torch.softmax(self.model(g["x"], g["ei"], g["ea"]), 1)[:, 1].numpy()
        s, d = find(df, SRC), find(df, DST)
        f = pd.DataFrame({"src": df[s].astype(str), "dst": df[d].astype(str), "p": prob})
        f["bad"] = f.p >= threshold
        out = f.groupby("src").agg(flows=("p", "size"), malicious=("bad", "sum"), peers=("dst", "nunique"), score=("p", "mean"))
        out["ratio"] = out.malicious / out.flows
        out["flagged"] = (out.malicious >= min_flows) & (out.ratio >= min_ratio)
        out = out.sort_values(["flagged", "score"], ascending=False)
        return f, out
