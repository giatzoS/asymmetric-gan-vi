import os
import json
import numpy as np
import torch

ROOT = "outputs_tik_sensitivity"
GAMMAS = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
SEEDS = [0, 1, 2]
ALPHA = 0.02


def ema(x, alpha=0.02):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def get_metric(m, *names):
    for name in names:
        if name in m:
            return float(m[name])
    return np.nan


print()
print("TIKHONOV GAMMA SENSITIVITY — CIFAR-10, Asym-Tik, seeds 0–2")
print("=" * 125)
print(
    f"{'gamma':>8} | "
    f"{'FID mean±std':>17} | "
    f"{'IS mean±std':>15} | "
    f"{'KID mean±std':>19} | "
    f"{'early':>9} | {'late':>9} | {'late/early':>10}"
)
print("-" * 125)

for gamma in GAMMAS:
    fids, kids, iss = [], [], []
    curves = []

    for seed in SEEDS:
        run = os.path.join(
            ROOT,
            f"gamma_{gamma}",
            "cifar10_dcgan32",
            f"asym_tik_seed{seed}",
        )

        with open(os.path.join(run, "metrics.json"), "r") as f:
            m = json.load(f)

        fids.append(get_metric(
            m,
            "frechet_inception_distance",
            "fid",
            "FID",
        ))

        kids.append(get_metric(
            m,
            "kernel_inception_distance_mean",
            "kid",
            "KID",
        ))

        iss.append(get_metric(
            m,
            "inception_score_mean",
            "is",
            "IS",
        ))

        x = torch.load(
            os.path.join(run, "update_grad_norm.pt"),
            map_location="cpu",
        )

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        curves.append(ema(x, ALPHA))

    # truncate to common length
    n = min(len(x) for x in curves)
    curves = np.stack([x[:n] for x in curves])

    # median across seeds at each step
    med = np.median(curves, axis=0)

    early_n = min(5000, n)
    late_n = min(5000, n)

    early = float(np.median(med[:early_n]))
    late = float(np.median(med[-late_n:]))
    ratio = late / early if early != 0 else np.nan

    fid_mean = np.nanmean(fids)
    fid_std = np.nanstd(fids, ddof=1)

    is_mean = np.nanmean(iss)
    is_std = np.nanstd(iss, ddof=1)

    kid_mean = np.nanmean(kids)
    kid_std = np.nanstd(kids, ddof=1)

    print(
        f"{gamma:>8} | "
        f"{fid_mean:6.2f} ± {fid_std:5.2f} | "
        f"{is_mean:5.2f} ± {is_std:4.2f} | "
        f"{kid_mean:7.4f} ± {kid_std:6.4f} | "
        f"{early:9.3g} | "
        f"{late:9.3g} | "
        f"{ratio:10.3f}"
    )

print("=" * 125)
