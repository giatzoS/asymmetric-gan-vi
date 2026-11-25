# make_results_table.py
import os
import json
from matplotlib.pylab import seed
import pandas as pd
import numpy as np
from collections import defaultdict

def main():
    out_dir = "./outputs"
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Missing outputs directory: {out_dir}")

    raw_rows = []

    for run_name in sorted(os.listdir(out_dir)):
        run_dir = os.path.join(out_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Parse method + seed (expected: <method>_seed<k>)
        if "_seed" in run_name:
            method = run_name.rsplit("_seed", 1)[0]
            seed_str = run_name.rsplit("_seed", 1)[1]
            try:
                seed = int(seed_str)
            except:
                seed = None
        else:
            method = run_name
            seed = None
        
        allowed_seeds = set(range(10))
        if seed is None or seed not in allowed_seeds:
            continue

        raw_rows.append({
            "run": run_name,
            "method": method,
            "seed": seed,
            "FID": metrics.get("frechet_inception_distance", None),
            "IS": metrics.get("inception_score_mean", None),
            "KID": metrics.get("kernel_inception_distance_mean", None),
        })

    if len(raw_rows) == 0:
        print("No metrics.json found. Run evaluate_all.py first.")
        return

    raw_df = pd.DataFrame(raw_rows)

    # Save raw per-seed table
    raw_csv = os.path.join(out_dir, "results_table_seeds_raw.csv")
    raw_df.to_csv(raw_csv, index=False)
    print(f"[OK] Saved per-seed results: {raw_csv}")

    # Aggregate across seeds per method
    agg_rows = []
    for method, group in raw_df.groupby("method"):
        fid_vals = group["FID"].dropna().astype(float).to_numpy()
        is_vals  = group["IS"].dropna().astype(float).to_numpy()
        kid_vals = group["KID"].dropna().astype(float).to_numpy()

        # ddof=1 => sample std; if n=1, std becomes NaN (that's fine)
        agg_rows.append({
            "Method": method,
            "n_seeds": int(group["seed"].nunique(dropna=True)),
            "FID_mean": float(np.mean(fid_vals)) if len(fid_vals) else np.nan,
            "FID_std":  float(np.std(fid_vals, ddof=1)) if len(fid_vals) > 1 else np.nan,
            "IS_mean":  float(np.mean(is_vals)) if len(is_vals) else np.nan,
            "IS_std":   float(np.std(is_vals, ddof=1)) if len(is_vals) > 1 else np.nan,
            "KID_mean": float(np.mean(kid_vals)) if len(kid_vals) else np.nan,
            "KID_std":  float(np.std(kid_vals, ddof=1)) if len(kid_vals) > 1 else np.nan,
        })

    agg_df = pd.DataFrame(agg_rows)

    # Sort by FID mean (lower is better)
    if agg_df["FID_mean"].notna().any():
        agg_df = agg_df.sort_values("FID_mean", ascending=True)

    # Create paper-friendly columns: mean ± std
    def fmt_pm(mu, sd, decimals=3):
        if pd.isna(mu):
            return "-"
        if pd.isna(sd):
            return f"{mu:.{decimals}f}"
        return f"{mu:.{decimals}f} ± {sd:.{decimals}f}"

    agg_df["FID"] = agg_df.apply(lambda r: fmt_pm(r["FID_mean"], r["FID_std"], decimals=2), axis=1)
    agg_df["IS"]  = agg_df.apply(lambda r: fmt_pm(r["IS_mean"],  r["IS_std"],  decimals=3), axis=1)
    agg_df["KID"] = agg_df.apply(lambda r: fmt_pm(r["KID_mean"], r["KID_std"], decimals=6), axis=1)

    paper_df = agg_df[["Method", "n_seeds", "FID", "IS", "KID"]]

    # Save aggregated
    agg_csv = os.path.join(out_dir, "results_table_aggregated.csv")
    paper_df.to_csv(agg_csv, index=False)
    print(f"[OK] Saved aggregated results: {agg_csv}")

    latex_path = os.path.join(out_dir, "results_table_aggregated_latex.txt")
    with open(latex_path, "w") as f:
        f.write(paper_df.to_latex(index=False))
    print(f"[OK] Saved LaTeX table: {latex_path}")

    print("\n=== PAPER TABLE (Aggregated over seeds) ===")
    print(paper_df.to_string(index=False))

if __name__ == "__main__":
    main()


