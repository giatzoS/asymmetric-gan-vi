"""Fresh-seed confirmation: ZC-GP (gamma 2) vs ZC-GP + SN (gamma 5), PlantVillage."""
import glob, json
import numpy as np
from scipy import stats

def load(method, gamma, key):
    paths = sorted(glob.glob(f"outputs_confirm/gamma_{gamma}/plantvillage_dcgan64/{method}_seed*/metrics.json"))
    return np.array([json.load(open(p))[key] for p in paths])

for key, name in [("frechet_inception_distance", "FID"), ("kernel_inception_distance_mean", "KID")]:
    a, b = load("zc_gp", "2.0", key), load("zc_gp_sn", "5.0", key)
    print(f"{name}: zc_gp(g=2) n={len(a)} {a.mean():.4g}+-{a.std(ddof=1):.3g} | zc_gp_sn(g=5) n={len(b)} {b.mean():.4g}+-{b.std(ddof=1):.3g}")
    t = stats.ttest_ind(b, a, equal_var=False)
    ci = t.confidence_interval(0.95)
    print(f"   SN - noSN = {b.mean() - a.mean():+.4g}, 95% CI [{ci.low:.4g}, {ci.high:.4g}], Welch p={t.pvalue:.3g}")
