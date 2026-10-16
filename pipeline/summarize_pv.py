import os
import json
import numpy as np

root = "outputs_final_pv30k/plantvillage_dcgan64"

methods = [
    "baseline_plain",
    "baseline_sn",
    "zc_gp",
    "zc_gp_sn",
    "asym_tik",
    "asym_tik_sn",
]

print("\nPER-SEED RESULTS")
print("=" * 70)

summary = []

for method in methods:
    fids = []
    kids = []

    for seed in range(10):
        path = os.path.join(root, f"{method}_seed{seed}", "metrics.json")

        with open(path) as f:
            m = json.load(f)

        fid = float(m["frechet_inception_distance"])
        kid = float(m["kernel_inception_distance_mean"])

        fids.append(fid)
        kids.append(kid)

        print(
            f"{method:18s} seed {seed}: "
            f"FID={fid:.4f}, KID={kid:.6f}"
        )

    fids = np.array(fids)
    kids = np.array(kids)

    summary.append((
        method,
        fids.mean(),
        fids.std(ddof=1),
        kids.mean(),
        kids.std(ddof=1),
    ))

print("\nFINAL 10-SEED RESULTS")
print("=" * 70)

summary.sort(key=lambda x: x[1])

for method, fid_mean, fid_std, kid_mean, kid_std in summary:
    print(
        f"{method:18s} "
        f"FID = {fid_mean:.2f} +/- {fid_std:.2f}    "
        f"KID = {kid_mean:.4f} +/- {kid_std:.4f}"
    )
