import os
import torch
import numpy as np
import matplotlib.pyplot as plt

root = "outputs_final_pv30k/plantvillage_dcgan64"

methods = [
    "baseline_plain",
    "baseline_sn",
    "zc_gp",
    "zc_gp_sn",
    "asym_tik",
    "asym_tik_sn",
]

labels = {
    "baseline_plain": "Baseline",
    "baseline_sn": "Baseline + SN",
    "zc_gp": "ZC-GP",
    "zc_gp_sn": "ZC-GP + SN",
    "asym_tik": "Asym-Tik",
    "asym_tik_sn": "Asym-Tik + SN",
}

def load(path):
    x = torch.load(path, map_location="cpu")
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float).reshape(-1)

def ema(x, alpha=0.02):
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

data = {}

print("\nDYNAMICS SUMMARY")
print("=" * 90)

for method in methods:
    curves = []

    for seed in range(10):
        path = os.path.join(
            root,
            f"{method}_seed{seed}",
            "update_grad_norm.pt"
        )
        curves.append(ema(load(path)))

    n = min(len(x) for x in curves)
    curves = np.stack([x[:n] for x in curves])

    med = np.median(curves, axis=0)
    q25 = np.percentile(curves, 25, axis=0)
    q75 = np.percentile(curves, 75, axis=0)

    data[method] = (med, q25, q75)

    early = np.median(med[:5000])
    mid = np.median(med[12500:17500])
    late = np.median(med[-5000:])
    peak = np.max(med)
    ratio = late / max(early, 1e-12)

    print(
        f"{method:18s} "
        f"early={early:12.4g} "
        f"mid={mid:12.4g} "
        f"late={late:12.4g} "
        f"late/early={ratio:8.3f} "
        f"peak={peak:12.4g}"
    )

plt.figure(figsize=(11, 6))

for method in methods:
    med, q25, q75 = data[method]
    steps = np.arange(len(med))
    plt.plot(steps, med, label=labels[method])
    plt.fill_between(steps, q25, q75, alpha=0.15)

plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("Alternating update-gradient norm proxy")
plt.legend()
plt.tight_layout()
plt.savefig("pv_dynamics.png", dpi=200)
plt.close()

print("\nSaved: pv_dynamics.png")
