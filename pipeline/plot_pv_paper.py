from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("./outputs_final_pv30k/plantvillage_dcgan64")
FIGDIR = Path("./figures")
FIGDIR.mkdir(exist_ok=True)

SEEDS = list(range(10))
EMA_ALPHA = 0.02
MAX_STEPS = 30000

# Larger paper-readable fonts
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})


def ema(x, alpha=0.02):
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    y[0] = x[0]

    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]

    return y


def load_method(method):
    curves = []

    for seed in SEEDS:
        path = ROOT / f"{method}_seed{seed}" / "update_grad_norm.pt"

        if not path.exists():
            raise FileNotFoundError(path)

        x = torch.load(path, map_location="cpu")

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

        x = np.asarray(x).squeeze()
        curves.append(ema(x, EMA_ALPHA))

    n = min(len(c) for c in curves)
    curves = np.stack([c[:n] for c in curves], axis=0)

    median = np.median(curves, axis=0)
    q25 = np.percentile(curves, 25, axis=0)
    q75 = np.percentile(curves, 75, axis=0)

    steps = np.linspace(1, MAX_STEPS, n)

    return steps, median, q25, q75


def make_plot(methods, outfile):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    eps = 1e-12

    for method, label in methods:
        steps, med, q25, q75 = load_method(method)

        line, = ax.plot(
            steps,
            np.maximum(med, eps),
            linewidth=2.0,
            label=label,
        )

        ax.fill_between(
            steps,
            np.maximum(q25, eps),
            np.maximum(q75, eps),
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_yscale("log")
    ax.set_xlim(0, MAX_STEPS)

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\widehat{G}_k$")

    ax.legend(
        frameon=True,
        fontsize=12,
        loc="best",
    )

    ax.grid(False)

    fig.tight_layout()
    fig.savefig(
        FIGDIR / outfile,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {FIGDIR / outfile}")


# (a) No spectral normalization
make_plot(
    [
        ("baseline_plain", "Baseline"),
        ("zc_gp", "ZC-GP"),
        ("asym_tik", "Asym-Tik"),
    ],
    "pv_dynamics_nosn.png",
)


# (b) With spectral normalization
make_plot(
    [
        ("baseline_sn", "Baseline + SN"),
        ("zc_gp_sn", "ZC-GP + SN"),
        ("asym_tik_sn", "Asym-Tik + SN"),
    ],
    "pv_dynamics_sn.png",
)
