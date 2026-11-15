# Botnet Watch - GNN botnet detection with a web interface

Flags hosts whose network behaviour looks like a botnet (DDoS floods, command-and-control beaconing),
using a graph neural network (GNN) in PyTorch, and shows them in a browser page. It is a **detection
assistant for human review**, not an automatic firewall (see "Limits").

## How it works
Each row of a host file is one host in one time window. Rows become graph nodes. Two kinds of edges link them:
the same host's consecutive windows, and each row's k most similar rows. A 2-layer GraphSAGE-style network
passes information along these edges and outputs a botnet probability per row. A host is flagged when enough
of its windows are suspicious (see the web page controls).

## Files
| File | Purpose |
|---|---|
| `train_hosts.py` | Train and evaluate the host GNN (main script) |
| `detect_hosts.py` | Loads a trained host model and scores a file |
| `server.py`, `static/index.html` | Web interface (Flask + plain HTML/JS) |
| `mitigation.py` | Blocking helper: prints `iptables` commands (dry run) unless enforcement is switched on |
| `baseline.py` | Non-graph gradient-boosting baseline for comparison |
| `inspect_windows.py` | Diagnostic: how the -in and -out files relate in time |
| `data.py`, `model.py`, `train.py`, `detect.py`, `make_demo_data.py` | Original version for connection logs (source IP / destination IP per flow), with fake demo data |
| `merge_inout.py` | Experimental, **not used**: the -in and -out files do not line up by SequenceId |

`data/` (your CSV files), `models/` (trained models) and `venv/` are not part of the code.

## Setup (Windows, Python 3.10+)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
If C: is short on space, point pip's temp folders at another drive first
(`$env:TMP`, `$env:TEMP`, `$env:PIP_CACHE_DIR`).

## Data format
Per-host CSV files (CTU-13 / NCC style) with columns `Address`, `SequenceId`, `Label` and numeric features.
Use the **-out** files: in the -in files each row is a single incoming connection and bots are very rare.
Any label other than normal / benign / background counts as botnet.

## Train
Honest evaluation (train on some scenarios, test on scenarios the model never saw):
```
$test  = Get-ChildItem data\ctu-scenario5-out.csv, data\ctu-scenario6-out.csv, data\ctu-scenario12-out.csv | % FullName
$train = Get-ChildItem data\ctu-scenario*-out.csv -Exclude ctu-scenario5-out.csv, ctu-scenario6-out.csv, ctu-scenario12-out.csv | % FullName
python train_hosts.py --data $train --test_data $test --normal_frac 0.2 --pos_weight 1 --epochs 20 --out models\host_gnn_eval.pt
```
`--normal_frac 0.2` keeps every bot row but only 20% of normal rows (saves memory, rebalances classes).

Final model (all scenarios; its own accuracy cannot be measured, use the figure above):
```
$all = Get-ChildItem data\ctu-scenario*-out.csv | % FullName
python train_hosts.py --data $all --normal_frac 0.2 --pos_weight 1 --epochs 20 --final --out models\host_gnn_final.pt
```

## Run the web interface
```
$env:HOST_MODEL = "models/host_gnn_final.pt"
python server.py
```
Open http://127.0.0.1:5000, upload a `-out` CSV and click Analyze. Settings that worked best:
window score above 0.7 to 0.9, flag a host after 5 suspicious windows (and at least 20% of its windows).
If the file has a `Label` column, the page shows how many flagged hosts were real bots.

## Results (held-out CTU scenarios 5, 6 and 12; 215,624 rows, 823 bot rows, only 3 bot hosts)
Row-level, single run:

| Threshold | GNN precision / recall / F1 | Plain gradient boosting precision / recall / F1 |
|---|---|---|
| 0.5 | 0.45 / 0.53 / 0.49 | 0.43 / 0.57 / 0.49 |
| 0.7 | 0.52 / 0.49 / 0.51 | 0.48 / 0.46 / 0.47 |
| 0.9 | 0.60 / 0.43 / 0.50 | 0.56 / 0.31 / 0.40 |
| 0.95 | 0.61 / 0.39 / 0.48 | 0.58 / 0.27 / 0.37 |

Summary: on botnets it has never seen, the model finds roughly half of bot activity, and roughly half of its
alarms are real. The GNN is about equal to the plain model when many false alarms are accepted and better when
few are wanted. Adding the NCC-2 files to training did not improve results.

## Limits
- Public, old datasets (CTU-13 is from 2011) and per-host summaries, not raw packets. Expect lower accuracy on modern traffic.
- Each CTU-13 scenario has a different botnet; results on new botnets are limited. The test covers 3 scenarios and 3 bot hosts, so every figure is rough and comes from one run.
- It finds suspicious hosts. It does not absorb a volumetric DDoS; that needs upstream protection (ISP, CDN) and rate limiting. Spoofed source addresses make per-IP blocking unreliable.
- Blocking is a dry run by default. Real blocking needs Linux and root. Keep a human review step, test on a lab network, and keep a whitelist, because a false positive can cut off a legitimate server.
- Best results should come from training on your own network's traffic.
- Only analyse networks you own or are authorised to monitor.
