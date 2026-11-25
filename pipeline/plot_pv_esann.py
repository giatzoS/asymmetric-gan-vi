# ESANN Fig. 1(b): all six PlantVillage configurations in one panel,
# same size/fonts/margins/colors as cifar10/plot_dynamics_esann.py
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("/data/P70093283/WCCIproj/avigan_refactored_v2/outputs_final_pv30k/plantvillage_dcgan64")
FIGDIR = Path("./figures"); FIGDIR.mkdir(exist_ok=True)
SEEDS = list(range(10))
EMA_ALPHA = 0.02
MAX_STEPS = 30000

# same order as the CIFAR-10 panel -> same default colors per method
METHODS = [
    ("baseline_plain", "Baseline"),
    ("baseline_sn", "Baseline + SN"),
    ("zc_gp", "ZC-GP"),
    ("zc_gp_sn", "ZC-GP + SN"),
    ("asym_tik", "Asym-Tik"),
    ("asym_tik_sn", "Asym-Tik + SN"),
]

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 5})

def ema(x, alpha):
    x = np.asarray(x, dtype=np.float64); y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y

def load_method(method):
    curves = []
    for seed in SEEDS:
        path = ROOT / f"{method}_seed{seed}" / "update_grad_norm.pt"
        x = torch.load(path, map_location="cpu")
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        curves.append(ema(np.asarray(x).squeeze(), EMA_ALPHA))
    n = min(len(c) for c in curves)
    A = np.stack([c[:n] for c in curves], axis=0)
    return (np.linspace(1, MAX_STEPS, n), np.median(A, 0),
            np.percentile(A, 25, 0), np.percentile(A, 75, 0))

fig = plt.figure(figsize=(2.4, 1.9))
eps = 1e-12
for method, label in METHODS:
    steps, med, q25, q75 = load_method(method)
    S = max(1, len(steps) // 2000)
    plt.plot(steps[::S], np.maximum(med[::S], eps), label=label, linewidth=0.9)
    plt.fill_between(steps[::S], np.maximum(q25[::S], eps), np.maximum(q75[::S], eps),
                     alpha=0.15, linewidth=0)
plt.yscale("log")
plt.xlim(0, MAX_STEPS)
plt.grid(True, which="both", linestyle="--", alpha=0.35)
plt.xlabel("Training step")
plt.ylabel(r"$\widehat{\mathcal{G}}_k$")
plt.legend(loc="lower left", ncol=2, fontsize=5, framealpha=0.40, handlelength=1.2,
           columnspacing=0.8, borderpad=0.3, labelspacing=0.2)
plt.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
plt.tick_params(axis="both", which="major", length=3)
plt.tick_params(axis="both", which="minor", length=1.5)
plt.subplots_adjust(left=0.18, right=0.97, bottom=0.20, top=0.97)
out = FIGDIR / "pv_dynamics_all.pdf"
plt.savefig(out); plt.close(fig)
print(f"[OK] Saved: {out}")
