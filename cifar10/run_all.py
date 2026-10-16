# run_all.py
import os
import torch
from train import train_one_case
from data import get_cifar10_loader

def get_cases():
    return {
        "baseline_plain": {
            "use_sn": False,
            "use_asym_gp": False,
            "use_tikhonov": False,
            "gamma_gp": 0.0,
            "gamma_tik": 0.0,
        },
        "baseline_sn": {
            "use_sn": True,
            "use_asym_gp": False,
            "use_tikhonov": False,
            "gamma_gp": 0.0,
            "gamma_tik": 0.0,
        },
        "proposed_GP_SN": {
            "use_sn": True,
            "use_asym_gp": True,
            "use_tikhonov": False,
            "gamma_gp": 0.5,
            "gamma_tik": 0.0,
        },
        "proposed_GP": {
            "use_sn": False,
            "use_asym_gp": True,
            "use_tikhonov": False,
            "gamma_gp": 0.5,
            "gamma_tik": 0.0,
        },
        "proposed_Tikh": {
            "use_sn": False,
            "use_asym_gp": False,
            "use_tikhonov": True,
            "gamma_gp": 0.0,
            "gamma_tik": 1e-6,
        },
        "proposed_Tikh_SN": {
            "use_sn": True,
            "use_asym_gp": False,
            "use_tikhonov": True,
            "gamma_gp": 0.0,
            "gamma_tik": 1e-6,
        }
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    cases = get_cases()

    seeds = [0, 1, 2, 3, 4]   # 5 seeds
    epochs = 150

    for seed in seeds:
        # ✅ make a seed-specific loader (shuffle order differs per seed, reproducible)
        loader = get_cifar10_loader(batch_size=64, device=device, seed=seed)

        for case_name, cfg in cases.items():
            run_name = f"{case_name}_seed{seed}"
            print(f"\n=== Running: {run_name} ===")

            train_one_case(
                case_name=run_name,
                cfg=cfg,
                loader=loader,
                device=device,
                out_dir=out_dir,
                epochs=epochs,
                save_every=500,
                seed=seed,
            )


if __name__ == "__main__":
    main()

