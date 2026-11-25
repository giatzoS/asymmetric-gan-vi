# ESANN Fig. 1: (a) CIFAR-10, (b) PlantVillage; one shared legend; Tikhonov curves at gamma_Tik = 1e-3.
from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

AVI = Path(os.environ.get("AVI", "/data/P70093283/WCCIproj/A-VI-GAN/outputs"))
PIPE = Path(os.environ.get("PIPE", "."))
PVM = Path(os.environ.get("PVM", "/data/P70093283/WCCIproj/avigan_refactored_v2/outputs_final_pv30k/plantvillage_dcgan64"))
OUT = Path(os.environ.get("FIG_OUT", "./figures/fig1_dynamics.pdf"))
SEEDS = range(10)
ALPHA = 0.02

C_T3 = PIPE / "outputs_asym_vs_sym_1e-3/cifar10_dcgan32"
T3 = PIPE / "outputs_tik1e3"

METHODS = [
    ("Baseline",      "C0", (AVI, "baseline_plain", "gaps.pt", None),   (PVM, "baseline_plain", "update_grad_norm.pt", 30000)),
    ("Baseline + SN", "C1", (AVI, "baseline_sn", "gaps.pt", None),      (PVM, "baseline_sn", "update_grad_norm.pt", 30000)),
    ("ZC-GP",         "C2", (AVI, "proposed_GP", "gaps.pt", None),      (PVM, "zc_gp", "update_grad_norm.pt", 30000)),
    ("ZC-GP + SN",    "C3", (AVI, "proposed_GP_SN", "gaps.pt", None),   (PVM, "zc_gp_sn", "update_grad_norm.pt", 30000)),
    ("Asym-Tik",      "C4", (C_T3, "asym_tik", "update_grad_norm.pt", 117300),
                            (T3 / "plantvillage_dcgan64", "asym_tik", "update_grad_norm.pt", 30000)),
    ("Asym-Tik + SN", "C5", (T3 / "cifar10_dcgan32", "asym_tik_sn", "update_grad_norm.pt", 117300),
                            (T3 / "plantvillage_dcgan64", "asym_tik_sn", "update_grad_norm.pt", 30000)),
    ("Sym-Tik",       "C6", (C_T3, "sym_tik", "update_grad_norm.pt", 117300), None),
    ("Sym-Tik + SN",  "C7", (T3 / "cifar10_dcgan32", "sym_tik_sn", "update_grad_norm.pt", 117300), None),
]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.labelsize": 7,
                     "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6})

def ema(x):
    x = np.asarray(x, dtype=np.float64).squeeze(); y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = ALPHA * x[i] + (1 - ALPHA) * y[i - 1]
    return y

def load(path):
    x = torch.load(path, map_location="cpu")
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def stats(root, prefix, fname):
    curves = [ema(load(root / f"{prefix}_seed{s}" / fname)) for s in SEEDS]
    n = min(len(c) for c in curves)
    A = np.stack([c[:n] for c in curves])
    return n, np.median(A, 0), np.percentile(A, 25, 0), np.percentile(A, 75, 0)

fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0))
handles = {}
for ax, title, col in [(axes[0], "(a) CIFAR-10", 2), (axes[1], "(b) PlantVillage", 3)]:
    xmax = 0
    for m in METHODS:
        label, color, spec = m[0], m[1], m[col]
        if spec is None:
            continue
        root, prefix, fname, steps = spec
        n, med, q25, q75 = stats(root, prefix, fname)
        x = np.arange(n) if steps is None else np.linspace(0, steps, n)
        S = max(1, n // 2000); x = x[::S] / 1000; xmax = max(xmax, x[-1])
        h, = ax.plot(x, np.maximum(med[::S], 1e-12), color=color, linewidth=0.8)
        ax.fill_between(x, np.maximum(q25[::S], 1e-12), np.maximum(q75[::S], 1e-12),
                        color=color, alpha=0.15, linewidth=0)
        handles[label] = h
        print(f"{title:18s} {label:14s} len={n} final_median={med[-1]:.3g}")
    ax.set_yscale("log"); ax.set_xlim(0, xmax)
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
