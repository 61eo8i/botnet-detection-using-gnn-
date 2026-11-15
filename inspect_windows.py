"""Shows how the -in and -out windows of one scenario relate in time.
Usage:  python inspect_windows.py ctu-scenario5"""
import sys
import pandas as pd

name = sys.argv[1] if len(sys.argv) > 1 else "ctu-scenario5"
ins = pd.read_csv(f"data/{name}-in.csv", low_memory=False)
outs = pd.read_csv(f"data/{name}-out.csv", low_memory=False)
for d in (ins, outs):
    d.columns = [c.strip() for c in d.columns]

common = set(ins.Address) & set(outs.Address)
print(f"hosts in the in-file: {ins.Address.nunique()}, in the out-file: {outs.Address.nunique()}, in both: {len(common)}")
if not common:
    sys.exit("No host appears in both files.")

# Pick the host that has the most windows across both files
counts = ins[ins.Address.isin(common)].groupby("Address").size().add(
    outs[outs.Address.isin(common)].groupby("Address").size(), fill_value=0)
host = counts.sort_values(ascending=False).index[0]
print(f"\nExample host: {host}\n")

for label, d, start, end, deg in (("IN windows", ins, "InStartTime", "InEndTime", "InDegree"),
                                  ("OUT windows", outs, "OutStartTime", "OutEndTime", "OutDegree")):
    cols = [c for c in ("SequenceId", start, end, deg, "Label") if c in d.columns]
    print(label)
    print(d[d.Address == host].sort_values(start)[cols].head(8).to_string(index=False), "\n")

print("Typical window size (all hosts):")
for label, d, start, end in (("in ", ins, "InStartTime", "InEndTime"), ("out", outs, "OutStartTime", "OutEndTime")):
    span = d[end] - d[start]
    gaps = d.sort_values(["Address", start]).groupby("Address")[start].diff().dropna()
    print(f"  {label}: median span of one window {span.median():.0f}s, 90th percentile {span.quantile(.9):.0f}s; "
          f"median gap between a host's consecutive windows {gaps.median():.0f}s")
