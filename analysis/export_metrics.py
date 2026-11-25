"""Collect per-run metrics.json files into one JSON file per study.

The resulting files in results/ let anyone recompute the tables and the
statistical tests without retraining or re-evaluating any model.

Usage (from the repository root, after training and evaluation):
  python analysis/export_metrics.py \
      --cifar-root cifar10/outputs \
      --pv-root pipeline/outputs_final_pv30k/plantvillage_dcgan64 \
      --tik-root pipeline/outputs_tik_sensitivity \
      --out-dir results
"""
import argparse
import glob
import json
import os

CIFAR_METHODS = [
    "baseline_plain", "baseline_sn", "proposed_GP", "proposed_GP_SN",
    "proposed_Tikh", "proposed_Tikh_SN", "symmetric_Tikh", "symmetric_Tikh_SN",
]
PV_METHODS = ["baseline_plain", "baseline_sn", "zc_gp", "zc_gp_sn", "asym_tik", "asym_tik_sn"]
MAIN_SEEDS = list(range(10))
TIK_SEEDS = [0, 1, 2]


def read_runs(root, methods, seeds):
    runs = {}
    for m in methods:
        runs[m] = {}
        for s in seeds:
            path = os.path.join(root, f"{m}_seed{s}", "metrics.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing metrics: {path}")
            with open(path) as f:
                runs[m][str(s)] = json.load(f)
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cifar-root", required=True)
    p.add_argument("--pv-root", required=True)
    p.add_argument("--tik-root", default=None)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    studies = {
        "metrics_cifar10.json": ("cifar10_dcgan32", read_runs(a.cifar_root, CIFAR_METHODS, MAIN_SEEDS), MAIN_SEEDS),
        "metrics_plantvillage.json": ("plantvillage_dcgan64", read_runs(a.pv_root, PV_METHODS, MAIN_SEEDS), MAIN_SEEDS),
    }
    if a.tik_root:
        tik = {}
        for d in sorted(glob.glob(os.path.join(a.tik_root, "gamma_*"))):
            gamma = os.path.basename(d).replace("gamma_", "")
            tik[gamma] = read_runs(os.path.join(d, "cifar10_dcgan32"), ["asym_tik"], TIK_SEEDS)["asym_tik"]
        studies["metrics_tik_sensitivity.json"] = ("cifar10_dcgan32 asym_tik, gamma sweep", tik, TIK_SEEDS)

    for fname, (study, runs, seeds) in studies.items():
        out = {"study": study, "seeds": seeds, "runs": runs}
        with open(os.path.join(a.out_dir, fname), "w") as f:
            json.dump(out, f, indent=1)
        print(f"[OK] {fname}")


if __name__ == "__main__":
    main()
