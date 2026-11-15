"""
merge_inout.py  -  join a scenario's "-in" and "-out" files into ONE file.

Each row of the result = one host in one time window, with BOTH views:
what the host received (in_...) and what it sent (out_...).

Usage:
  python merge_inout.py --name ctu-scenario5
  python merge_inout.py --name ctu-scenario3 --normal_frac 0.2
      (keeps every botnet row but only 20% of normal rows - use for training files)

Output: data/merged/<name>-both.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

# Label values that mean "not a bot"
BENIGN_LABELS = {"0", "normal", "benign", "background", "none", "nan", ""}


def is_bot(labels):
    """Given a column of labels, return True where the label means botnet."""
    text = labels.astype(str).str.strip().str.lower()
    not_bot = text.isin(BENIGN_LABELS) | text.str.contains("background")
    return ~not_bot


def find_bot_ids(path):
    """Quickly scan a file and collect the SequenceId of every botnet row."""
    bot_ids = set()
    only_needed_columns = lambda col: col.strip() in ("SequenceId", "Label")
    for chunk in pd.read_csv(path, usecols=only_needed_columns, chunksize=1_000_000):
        chunk.columns = [c.strip() for c in chunk.columns]
        bot_ids |= set(chunk.loc[is_bot(chunk["Label"]), "SequenceId"])
    return bot_ids


def load_file(path, bot_ids, normal_frac, chunk_size=500_000):
    """Read a CSV in pieces (saves memory).
    If normal_frac < 1, keep all botnet rows plus that fraction of the others.
    The choice depends only on SequenceId, so the in-file and out-file keep matching rows."""
    kept_parts = []
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk.columns = [c.strip() for c in chunk.columns]
        if normal_frac < 1.0:
            hashes = pd.util.hash_pandas_object(chunk["SequenceId"].astype(str), index=False).values
            picked_by_hash = (hashes % 1000) < int(normal_frac * 1000)
            is_bot_row = chunk["SequenceId"].isin(bot_ids).values
            chunk = chunk[picked_by_hash | is_bot_row]
        kept_parts.append(chunk)
    return pd.concat(kept_parts, ignore_index=True)


def add_prefix(df, prefix):
    """Rename columns so the in and out files don't clash: MeanDur -> in_MeanDur / out_MeanDur.
    Address and Label become Address_in / Address_out and Label_in / Label_out.
    SequenceId keeps its name because we join on it."""
    new_names = {}
    for col in df.columns:
        if col == "SequenceId":
            new_names[col] = col
        elif col in ("Address", "Label"):
            new_names[col] = col + "_" + prefix
        else:
            new_names[col] = prefix + "_" + col
    return df.rename(columns=new_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="e.g. ctu-scenario5")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="data/merged")
    parser.add_argument("--normal_frac", type=float, default=1.0)
    args = parser.parse_args()

    in_path = os.path.join(args.data_dir, f"{args.name}-in.csv")
    out_path = os.path.join(args.data_dir, f"{args.name}-out.csv")

    # 1. Load both files (optionally sampling the normal rows)
    if args.normal_frac < 1.0:
        bot_ids = find_bot_ids(in_path) | find_bot_ids(out_path)
    else:
        bot_ids = set()
    in_df = load_file(in_path, bot_ids, args.normal_frac)
    out_df = load_file(out_path, bot_ids, args.normal_frac)

    # 2. Drop columns we don't need, and duplicate window ids
    in_df = in_df.drop(columns=["Dst-Id"], errors="ignore").drop_duplicates("SequenceId")
    out_df = out_df.drop(columns=["Src-Id"], errors="ignore").drop_duplicates("SequenceId")

    # 3. Join on SequenceId (keep rows that exist in only one of the files)
    merged = add_prefix(in_df, "in").merge(add_prefix(out_df, "out"), on="SequenceId", how="outer")

    # 4. Print how well the two files matched
    in_present = merged["Address_in"].notna()
    out_present = merged["Address_out"].notna()
    matched = in_present & out_present
    print(f"{args.name}: in rows {len(in_df)}, out rows {len(out_df)}, matched {int(matched.sum())}, "
          f"only-in {int((~out_present).sum())}, only-out {int((~in_present).sum())}")

    # 5. Check the two files agree on timing (small gap = good)
    if "in_InStartTime" in merged and "out_OutStartTime" in merged and matched.any():
        in_time = pd.to_numeric(merged["in_InStartTime"])
        out_time = pd.to_numeric(merged["out_OutStartTime"])
        gap = (in_time - out_time).abs()[matched]
        print(f"median gap between in/out start times of matched rows: {gap.median():.1f} seconds")
        merged["StartTime"] = out_time.fillna(in_time)

    # 6. Build the final columns
    merged["has_in"] = in_present.astype(int)      # 1 if the host has an "in" record for this window
    merged["has_out"] = out_present.astype(int)    # 1 if the host has an "out" record for this window
    merged["Address"] = merged["Address_out"].fillna(merged["Address_in"])
    bot_in = is_bot(merged["Label_in"].fillna("normal"))
    bot_out = is_bot(merged["Label_out"].fillna("normal"))
    merged["Label"] = np.where(bot_in | bot_out, "botnet", "normal")   # bot if either file says so

    feature_cols = [c for c in merged.columns if c.startswith(("in_", "out_"))]
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    front = [c for c in ("Address", "SequenceId", "Label", "StartTime", "has_in", "has_out") if c in merged.columns]

    # 7. Save
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"{args.name}-both.csv")
    merged[front + feature_cols].to_csv(save_path, index=False)
    n_bots = int((merged["Label"] == "botnet").sum())
    print(f"saved {save_path}: {len(merged)} rows, {n_bots} botnet rows, {len(feature_cols) + 2} feature columns")


if __name__ == "__main__":
    main()