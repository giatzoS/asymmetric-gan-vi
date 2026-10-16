"""
Pairwise statistical comparison of per-seed FID/KID for the CIFAR-10 and
PlantVillage studies.

For every comparison it reports:
  - mean +/- std (ddof=1) of both groups
  - Welch's two-sided t-test (primary; does not assume equal variances)
  - Holm-adjusted Welch p-value over all comparisons of the same metric
  - Mann-Whitney U two-sided test (robustness; no normality assumption)
  - paired t-test over matched seeds (robustness; same seed = same init/data order)

Metrics are read either from the published JSON files in results/ (default,
no training needed) or from the per-run metrics.json files after training.

Usage (from the repository root):
  python analysis/stats_tests.py                         # uses results/*.json
  python analysis/stats_tests.py --metric kid
  python analysis/stats_tests.py --source runs \
      --cifar-root cifar10/outputs \
      --pv-root pipeline/outputs_final_pv30k/plantvillage_dcgan64
  python analysis/stats_tests.py --out results/stats/stats_tests_fid.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel

SEEDS = list(range(10))

METRIC_KEYS = {
    "fid": "frechet_inception_distance",
    "kid": "kernel_inception_distance_mean",
}

COMPARISONS = {
    "CIFAR-10": [
        ("proposed_GP", "baseline_sn"),
        ("proposed_GP", "proposed_GP_SN"),
        ("baseline_sn", "baseline_plain"),
        ("proposed_Tikh", "baseline_plain"),
        ("proposed_Tikh_SN", "baseline_sn"),
        ("proposed_Tikh", "symmetric_Tikh"),
        ("proposed_Tikh_SN", "symmetric_Tikh_SN"),
    ],
    "PlantVillage": [
        ("zc_gp", "baseline_sn"),
        ("zc_gp", "zc_gp_sn"),
        ("baseline_sn", "baseline_plain"),
        ("asym_tik", "baseline_plain"),
        ("asym_tik_sn", "baseline_sn"),
        ("asym_tik_sn", "zc_gp_sn"),
    ],
}

JSON_FILES = {"CIFAR-10": "metrics_cifar10.json", "PlantVillage": "metrics_plantvillage.json"}


def make_loader(args):
    if args.source == "json":
        data = {}
        for ds, fname in JSON_FILES.items():
            with open(os.path.join(args.results_dir, fname)) as f:
                data[ds] = json.load(f)["runs"]

        def load(dataset, method, key):
            return np.array([float(data[dataset][method][str(s)][key]) for s in SEEDS])
    else:
        roots = {"CIFAR-10": args.cifar_root, "PlantVillage": args.pv_root}

        def load(dataset, method, key):
            vals = []
            for s in SEEDS:
                path = os.path.join(roots[dataset], f"{method}_seed{s}", "metrics.json")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing metrics: {path}")
                with open(path) as f:
                    vals.append(float(json.load(f)[key]))
            return np.array(vals)
    return load


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=list(METRIC_KEYS), default="fid")
    parser.add_argument("--source", choices=["json", "runs"], default="json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--cifar-root", default="cifar10/outputs")
    parser.add_argument("--pv-root", default="pipeline/outputs_final_pv30k/plantvillage_dcgan64")
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    args = parser.parse_args()
    key = METRIC_KEYS[args.metric]
    load = make_loader(args)

    rows = []
    for dataset, pairs in COMPARISONS.items():
        for a, b in pairs:
            xa, xb = load(dataset, a, key), load(dataset, b, key)
            rows.append({
                "dataset": dataset,
                "A": a,
                "B": b,
                "n": f"{len(xa)},{len(xb)}",
                "mean_A": xa.mean(),
                "std_A": xa.std(ddof=1),
                "mean_B": xb.mean(),
                "std_B": xb.std(ddof=1),
                "diff_A_minus_B": xa.mean() - xb.mean(),
                "p_welch": ttest_ind(xa, xb, equal_var=False).pvalue,
                "p_mannwhitney": mannwhitneyu(xa, xb, alternative="two-sided").pvalue,
                "p_paired": ttest_rel(xa, xb).pvalue,
            })

    df = pd.DataFrame(rows)
    df["p_welch_holm"] = holm(df["p_welch"].values)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    num = "{:.4f}" if args.metric == "kid" else "{:.2f}"
    fmt = {c: (lambda v: f"{v:.4g}  ") for c in ["p_welch", "p_welch_holm", "p_mannwhitney", "p_paired"]}
    fmt.update({c: (lambda v, n=num: n.format(v)) for c in ["mean_A", "std_A", "mean_B", "std_B", "diff_A_minus_B"]})
    print(f"\nPairwise comparisons on {args.metric.upper()} ({len(SEEDS)} seeds per method)\n")
    print(df.to_string(index=False, formatters=fmt))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
