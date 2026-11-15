"""Synthetic flows so you can smoke-test the pipeline before using a real dataset."""
import numpy as np, pandas as pd
rng = np.random.default_rng(1); rows = []
bots = [f"10.0.9.{i}" for i in range(1, 15)]; c2 = "203.0.113.7"; victim = "10.0.0.5"
users = [f"10.0.1.{i}" for i in range(1, 80)]; servers = [f"198.51.100.{i}" for i in range(1, 30)]
for _ in range(40000):
    if rng.random() < .15:   # botnet: beaconing to C2 + DDoS flood on victim
        s = rng.choice(bots); d = c2 if rng.random() < .4 else victim
        r = [rng.integers(1, 4), rng.integers(40, 120), rng.random() * .05, rng.integers(1, 3), 1]
    else:
        s = rng.choice(users); d = rng.choice(servers)
        r = [rng.integers(5, 400), rng.integers(500, 90000), rng.random() * 30, rng.integers(5, 300), 0]
    rows.append([s, d, *r])
pd.DataFrame(rows, columns=["Src IP", "Dst IP", "Packets", "Bytes", "Duration", "Dst Port Count", "Label"]).sample(frac=1, random_state=1).to_csv("data/demo.csv", index=False)
print("wrote data/demo.csv")
