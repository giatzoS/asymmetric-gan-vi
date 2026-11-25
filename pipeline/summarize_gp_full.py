"""gamma_GP sweep summary for ZC-GP and ZC-GP + SN (run from pipeline/)."""
import glob, json, os
import numpy as np
from scipy import stats

FID = "frechet_inception_distance"
GAMMAS = ["0.05", "0.2", "0.5", "2.0", "5.0", "10.0"]
REF = "../results/metrics_plantvillage.json"
ref = json.load(open(REF))["runs"] if os.path.exists(REF) else {}


def runs(pattern):
    out = {}
    for p in glob.glob(pattern):
        out[int(p.split("_seed")[1].split("/")[0])] = json.load(open(p))[FID]
    return out


for name, root, exp in [("PlantVillage, 30k steps", "outputs_gp_sweep", "plantvillage_dcgan64"),
                        ("CIFAR-10, 30k steps", "outputs_gp_sweep_c10", "cifar10_dcgan32")]:
    print(f"\n== {name}: FID mean +- std (n) per gamma_GP ==")
    best = {}
    for m in ["zc_gp", "zc_gp_sn"]:
        cells = []
        for g in GAMMAS:
            r = runs(f"{root}/gamma_{g}/{exp}/{m}_seed*/metrics.json")
            if not r and exp == "plantvillage_dcgan64" and g == "0.5" and m in ref:
                r = {int(k): v[FID] for k, v in ref[m].items() if int(k) in (0, 1, 2)}
            if not r:
                continue
            f = np.array(list(r.values()))
            sd = f.std(ddof=1) if len(f) > 1 else float("nan")
            cells.append(f"{g}: {f.mean():.2f}+-{sd:.2f} (n={len(f)})")
            if len(f) > 1 and (m not in best or f.mean() < best[m][1].mean()):
                best[m] = (g, f)
        print(f"{m:<9} " + " | ".join(cells))
    if len(best) == 2:
        (ga, a), (gb, b) = best["zc_gp"], best["zc_gp_sn"]
        t = stats.ttest_ind(b, a, equal_var=False)
        ci = t.confidence_interval(0.95)
        print(f"best zc_gp_sn (gamma={gb}) - best zc_gp (gamma={ga}): {b.mean() - a.mean():+.2f} FID, "
              f"95% CI [{ci.low:.2f}, {ci.high:.2f}], Welch p={t.pvalue:.3g}")
