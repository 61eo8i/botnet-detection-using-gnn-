"""Turns a flow CSV (CTU-13, CIC-IDS2017, UNSW-NB15, Bot-IoT, ...) into graphs.
Node = IP address. Edge = one network flow (carries the flow's features)."""
import numpy as np, pandas as pd, torch

SRC = ["src_ip", "source_ip", "srcip", "ipv4_src_addr", "saddr", "srcaddr", "sa"]
DST = ["dst_ip", "destination_ip", "dstip", "ipv4_dst_addr", "daddr", "dstaddr", "da"]
LAB = ["label", "class", "attack", "attack_cat", "target", "category"]
BENIGN = {"0", "benign", "normal", "background", "none", "nan", ""}

def clean_cols(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def find(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def to_binary(v):
    s = str(v).strip().lower()
    if s in BENIGN or s.startswith("flow=from-normal") or s.startswith("flow=background") or "background" in s:
        return 0
    return 1

def load_csv(path, max_rows=None):
    df = clean_cols(pd.read_csv(path, nrows=max_rows, low_memory=False))
    return df

def prepare(df, scaler=None):
    """Returns (df, X float32 [N,F], y int [N] or None, scaler dict)."""
    s, d, l = find(df, SRC), find(df, DST), find(df, LAB)
    if s is None or d is None:
        raise ValueError("CSV needs source & destination IP columns (e.g. 'Src IP', 'Dst IP').")
    y = df[l].map(to_binary).values.astype(np.int64) if l else None
    drop = {s, d, l, "flow_id", "timestamp", "stime", "starttime"}
    if scaler is None:
        num = df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes(include=[np.number])
        cols = list(num.columns)
    else:
        cols = scaler["cols"]
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float64)
    X = np.sign(X) * np.log1p(np.abs(X))            # tame heavy tails (bytes, durations)
    if scaler is None:
        scaler = {"cols": cols, "mean": X.mean(0), "std": X.std(0) + 1e-6}
    X = ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)
    return df, X, y, scaler

def build_graphs(df, X, y, window=5000):
    """Slice flows (time order) into windows; each window becomes one graph."""
    s, d = find(df, SRC), find(df, DST)
    graphs = []
    for a in range(0, len(df), window):
        b = min(a + window, len(df))
        src, dst = df[s].astype(str).values[a:b], df[d].astype(str).values[a:b]
        ips, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
        n = b - a
        g = {"ips": ips, "idx": np.arange(a, b),
             "ei": torch.tensor(np.stack([inv[:n], inv[n:]]), dtype=torch.long),
             "ea": torch.tensor(X[a:b]),
             "x": torch.ones(len(ips), X.shape[1]),   # nodes start identical; edges carry the signal
             "y": torch.tensor(y[a:b]) if y is not None else None}
        graphs.append(g)
    return graphs
