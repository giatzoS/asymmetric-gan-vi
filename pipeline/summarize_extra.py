"""Summarize the extra ESANN runs (run from pipeline/)."""
import glob, json, os
import numpy as np
from scipy import stats

FID, KID = "frechet_inception_distance", "kernel_inception_distance_mean"


def load(pattern):
    out = {}
    for d in sorted(glob.glob(pattern)):
        p = os.path.join(d, "metrics.json")
        if os.path.exists(p):
            seed = int(d.rsplit("seed", 1)[1])
            out[seed] = json.load(open(p))
    return out


def line(name, runs):
    if not runs:
        print(f"{name:<28} (no finished runs)")
        return None
    f = np.array([runs[s][FID] for s in sorted(runs)])
    k = np.array([runs[s][KID] for s in sorted(runs)])
    sd = lambda x: x.std(ddof=1) if len(x) > 1 else float("nan")
    print(f"{name:<28} n={len(f):<2} FID {f.mean():7.2f} +- {sd(f):5.2f}   KID {k.mean():.4f} +- {sd(k):.4f}")
    return f


print("\n== Asym-Tik vs Sym-Tik, gamma=1e-3, CIFAR-10 ==")
root = "outputs_asym_vs_sym_1e-3/cifar10_dcgan32"
a = line("asym_tik", load(f"{root}/asym_tik_seed*"))
s = line("sym_tik", load(f"{root}/sym_tik_seed*"))
if a is not None and s is not None and len(a) > 1 and len(s) > 1:
    t = stats.ttest_ind(a, s, equal_var=False)
    ci = t.confidence_interval(0.95)
    print(f"FID difference asym - sym: {a.mean() - s.mean():+.2f}  95% CI [{ci.low:.2f}, {ci.high:.2f}]  Welch p={t.pvalue:.3g}")

print("\n== gamma_GP sweep, ZC-GP + SN, PlantVillage, seeds 0-2 ==")
for g in ["0.05", "0.2", "2.0"]:
    line(f"zc_gp_sn  gamma_gp={g}", load(f"outputs_gp_sweep/gamma_{g}/plantvillage_dcgan64/zc_gp_sn_seed*"))
ref = "../results/metrics_plantvillage.json"
if os.path.exists(ref):
    r = json.load(open(ref))["runs"]
    for m in ["zc_gp_sn", "zc_gp"]:
        line(f"{m}  gamma_gp=0.5 (paper)", {int(k): v for k, v in r[m].items() if int(k) in (0, 1, 2)})
