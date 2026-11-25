"""Collect per-run metrics.json files into results/all_runs.csv and results/statistics.md.
statistics.md: mean +- std (ddof=1) per configuration and, within each (label, experiment),
two-sided Welch t-tests for all pairs (FID, KID), 95% CIs and Holm-adjusted p-values.
"""
import sys, re, json, csv, itertools
from pathlib import Path
import numpy as np
from scipy import stats

KEYS = [("frechet_inception_distance", "FID"),
        ("inception_score_mean", "IS"),
        ("kernel_inception_distance_mean", "KID")]
OUT = Path("results")
OUT.mkdir(exist_ok=True)

rows = []
for arg in sys.argv[1:]:
    label, root = arg.split("=", 1)
    for p in sorted(Path(root).rglob("metrics.json")):
        m = re.match(r"(.+)_seed(\d+)$", p.parent.name)
        if not m:
            continue
        d = json.load(open(p))
        r = {"label": label, "experiment": p.parent.parent.name,
             "method": m.group(1), "seed": int(m.group(2))}
        for k, n in KEYS:
            r[n] = d.get(k, "")
        rows.append(r)

with open(OUT / "all_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label", "experiment", "method", "seed", "FID", "IS", "KID"])
    w.writeheader()
    w.writerows(rows)


def vals(rs, n):
    return np.array([r[n] for r in rs if r[n] != ""], dtype=float)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


groups = {}
for r in rows:
    groups.setdefault((r["label"], r["experiment"]), {}).setdefault(r["method"], []).append(r)

L = ["# Statistics", "",
     "Mean ± std (ddof=1) over seeds; Welch two-sided t-tests, 95% CI of A − B, "
     "Holm correction within each table."]

for (label, exp), meths in sorted(groups.items()):
    L += ["", f"## {label} / {exp}", "",
          "| method | n | FID | IS | KID |", "|---|---|---|---|---|"]
    for meth, rs in sorted(meths.items()):
        cells = []
        for _, n in KEYS:
            v = vals(rs, n)
            cells.append(f"{v.mean():.4g} ± {v.std(ddof=1):.2g}" if len(v) > 1 else "–")
        L.append(f"| {meth} | {len(rs)} | " + " | ".join(cells) + " |")

    for metric in ["FID", "KID"]:
        res = []
        for a, b in itertools.combinations(sorted(meths), 2):
            va, vb = vals(meths[a], metric), vals(meths[b], metric)
            if len(va) < 2 or len(vb) < 2:
                continue
            p = stats.ttest_ind(va, vb, equal_var=False).pvalue
            sa, sb = va.var(ddof=1) / len(va), vb.var(ddof=1) / len(vb)
            dof = (sa + sb) ** 2 / (sa ** 2 / (len(va) - 1) + sb ** 2 / (len(vb) - 1))
            half = stats.t.ppf(0.975, dof) * np.sqrt(sa + sb)
            d = va.mean() - vb.mean()
            res.append((a, b, d, d - half, d + half, p))
        if not res:
            continue
        adj = holm(np.array([x[5] for x in res]))
        L += ["", f"**Welch tests, {metric}**", "",
              "| A | B | A − B | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|"]
        for (a, b, d, lo, hi, p), pa in zip(res, adj):
            L.append(f"| {a} | {b} | {d:+.4g} | [{lo:.4g}, {hi:.4g}] | {p:.2g} | {pa:.2g} |")

(OUT / "statistics.md").write_text("\n".join(L) + "\n")
print(f"{len(rows)} runs -> results/all_runs.csv, results/statistics.md")