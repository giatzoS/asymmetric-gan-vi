# ESANN Fig. 1: (a) CIFAR-10 and (b) PlantVillage side by side, one shared legend below.
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

CIFAR = Path(os.environ.get("CIFAR_OUT", "/data/P70093283/WCCIproj/A-VI-GAN/outputs"))
PV = Path(os.environ.get("PV_OUT", "/data/P70093283/WCCIproj/avigan_refactored_v2/outputs_final_pv30k/plantvillage_dcgan64"))
OUT = Path(os.environ.get("FIG_OUT", "./figures/fig1_dynamics.pdf"))
SEEDS = range(10)
ALPHA = 0.02

# label -> (color, CIFAR folder prefix, PV folder prefix); same color in both panels
METHODS = [
    ("Baseline",      "C0", "baseline_plain",    "baseline_plain"),
    ("Baseline + SN", "C1", "baseline_sn",       "baseline_sn"),
    ("ZC-GP",         "C2", "proposed_GP",       "zc_gp"),
    ("ZC-GP + SN",    "C3", "proposed_GP_SN",    "zc_gp_sn"),
    ("Asym-Tik",      "C4", "proposed_Tikh",     "asym_tik"),
    ("Asym-Tik + SN", "C5", "proposed_Tikh_SN",  "asym_tik_sn"),
    ("Sym-Tik",       "C6", "symmetric_Tikh",    None),
    ("Sym-Tik + SN",  "C7", "symmetric_Tikh_SN", None),
]

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6})

def ema(x):
    x = np.asarray(x, dtype=np.float64).squeeze(); y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = ALPHA * x[i] + (1 - ALPHA) * y[i - 1]
    return y

def load(path):
    x = torch.load(path, map_location="cpu")
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def stats(files):
    curves = [ema(load(f)) for f in files]
    n = min(len(c) for c in curves)
    A = np.stack([c[:n] for c in curves])
    return n, np.median(A, 0), np.percentile(A, 25, 0), np.percentile(A, 75, 0)

fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0))
handles = {}
for ax, title, root, fname, col, max_steps in [
        (axes[0], "(a) CIFAR-10", CIFAR, "gaps.pt", 2, None),
        (axes[1], "(b) PlantVillage", PV, "update_grad_norm.pt", 3, 30000)]:
    for m in METHODS:
        label, color, prefix = m[0], m[1], m[col]
        if prefix is None:
            continue
        n, med, q25, q75 = stats([root / f"{prefix}_seed{s}" / fname for s in SEEDS])
        x = np.arange(n) if max_steps is None else np.linspace(1, max_steps, n)
        S = max(1, n // 2000); x = x[::S] / 1000
        h, = ax.plot(x, np.maximum(med[::S], 1e-12), color=color, linewidth=0.8)
        ax.fill_between(x, np.maximum(q25[::S], 1e-12), np.maximum(q75[::S], 1e-12),
                        color=color, alpha=0.15, linewidth=0)
        handles[label] = h
    ax.set_yscale("log")
    ax.set_xlim(0, x[-1])
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.set_xlabel(r"Training step ($\times 10^3$)")
    ax.set_title(title, pad=3)
    ax.tick_params(which="major", length=3); ax.tick_params(which="minor", length=1.5)
axes[0].set_ylabel(r"$\widehat{\mathcal{G}}_k$")

labels = [m[0] for m in METHODS]
fig.legend([handles[l] for l in labels], labels, loc="lower center", ncol=4,
           frameon=False, handlelength=1.5, columnspacing=1.2, labelspacing=0.3)
fig.subplots_adjust(left=0.09, right=0.99, top=0.915, bottom=0.345, wspace=0.22)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print(f"[OK] Saved: {OUT}")
