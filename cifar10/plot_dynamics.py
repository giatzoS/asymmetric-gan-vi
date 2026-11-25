# plot_dynamics.py
import os
import re
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

SEED_RE = re.compile(r"(.+)_seed(\d+)$")

# --- Plot-only display names (DO NOT change base_cases / run folder names) ---
DISPLAY_NAME = {
    "proposed_GP": "ZC-GP",
    "proposed_GP_SN": "ZC-GP + SN",
    "proposed_Tikh": "Asym-Tik",
    "proposed_Tikh_SN": "Asym-Tik + SN",
    "symmetric_Tikh": "Sym-Tik",
    "symmetric_Tikh_SN": "Sym-Tik + SN",
    # baselines left as-is by default; include overrides here if you ever want:
    "baseline_plain": "Baseline",
    "baseline_sn": "Baseline + SN",
}

def display_case_name(base: str) -> str:
    return DISPLAY_NAME.get(base, base)

def load_gaps(path):
    if not os.path.exists(path):
        return None
    gaps = torch.load(path, map_location="cpu")
    if torch.is_tensor(gaps):
        gaps = gaps.cpu().numpy()
    else:
        gaps = np.array(gaps)
    return gaps

def ema_smooth(x, alpha=0.02):
    """Exponential moving average smoothing."""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def collect_seed_runs(out_dir, base_case):
    """
    Collect all runs in out_dir of the form:
      {base_case}_seed{seed}
    Returns list of (seed, run_dir).
    """
    runs = []
    for name in os.listdir(out_dir):
        m = SEED_RE.match(name)
        if not m:
            continue
        case, seed = m.group(1), int(m.group(2))
        if case == base_case:
            runs.append((seed, os.path.join(out_dir, name)))
    runs.sort(key=lambda t: t[0])
    return runs

def align_by_min_length(arrays):
    """Trim all arrays to the minimum length so we can stack safely."""
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays], min_len

def plot_cases_iqr(out_dir, base_cases, save_path, max_steps=None, ema_alpha=0.02):
    # --- Bigger fonts everywhere ---
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    plt.figure(figsize=(8.5, 4.2))
    eps = 1e-12  # avoid log(0) if ever happens

    for base in base_cases:
        runs = collect_seed_runs(out_dir, base)
        if len(runs) == 0:
            print(f"[SKIP] No runs found for base case: {base}")
            continue

        per_seed = []
        for seed, run_dir in runs:
            gaps_path = os.path.join(run_dir, "gaps.pt")
            gaps = load_gaps(gaps_path)
            if gaps is None:
                print(f"[SKIP] Missing {gaps_path}")
                continue

            if max_steps is not None:
                gaps = gaps[:max_steps]

            gaps_s = ema_smooth(gaps, alpha=ema_alpha)
            per_seed.append(gaps_s)

        if len(per_seed) == 0:
            print(f"[SKIP] No gaps loaded for: {base}")
            continue

        per_seed, L = align_by_min_length(per_seed)
        A = np.stack(per_seed, axis=0)  # (n_seeds, L)

        med = np.median(A, axis=0)
        q25 = np.percentile(A, 25, axis=0)
        q75 = np.percentile(A, 75, axis=0)

        med = np.maximum(med, eps)
        q25 = np.maximum(q25, eps)
        q75 = np.maximum(q75, eps)

        label = display_case_name(base)
        plt.plot(med, label=label, linewidth=2.0)
        plt.fill_between(np.arange(L), q25, q75, alpha=0.15)

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)

    # --- No title ---
    # plt.title(...)

    plt.xlabel("Training step")

    # --- Y label exactly as requested ---
    plt.ylabel(r"$\widehat{G}_k$")

    plt.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.40)

    plt.tick_params(axis="both", which="major", length=6)
    plt.tick_params(axis="both", which="minor", length=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--ema_alpha", type=float, default=0.02)
    args = parser.parse_args()

    base_cases = [
        "baseline_plain",
        "baseline_sn",
        "proposed_GP",
        "proposed_GP_SN",
        "proposed_Tikh",
        "proposed_Tikh_SN",
        "symmetric_Tikh",
        "symmetric_Tikh_SN",
    ]

    plot_cases_iqr(
        out_dir=args.out_dir,
        base_cases=base_cases,
        save_path=os.path.join(args.out_dir, "dynamics_all_cases.png"),
        max_steps=None,
        ema_alpha=args.ema_alpha,
    )

    plot_cases_iqr(
        out_dir=args.out_dir,
        base_cases=base_cases,
        save_path=os.path.join(args.out_dir, "dynamics_early.png"),
        max_steps=5000,
        ema_alpha=args.ema_alpha,
    )

if __name__ == "__main__":
    main()
