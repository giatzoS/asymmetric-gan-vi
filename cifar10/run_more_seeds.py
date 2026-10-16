import argparse
import os
import torch

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--group", choices=["main", "sym", "sym_sn"], required=True)
p.add_argument("--out_dir", default="./outputs")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out_dir, exist_ok=True)
seed = args.seed


def done(name):
    return os.path.exists(os.path.join(args.out_dir, name, "G_final.pt"))


if args.group == "main":
    from run_all import get_cases
    from train import train_one_case
    from data import get_cifar10_loader

    loader = get_cifar10_loader(batch_size=64, device=device, seed=seed)
    for case_name, cfg in get_cases().items():
        run_name = f"{case_name}_seed{seed}"
        if done(run_name):
            print(f"[SKIP] {run_name}")
            continue
        print(f"\n=== Running: {run_name} ===")
        train_one_case(case_name=run_name, cfg=cfg, loader=loader, device=device,
                       out_dir=args.out_dir, epochs=150, save_every=500, seed=seed)

elif args.group == "sym":
    from run_symmetric_tikh import train_symmetric_tikh_run
    run_name = f"symmetric_Tikh_seed{seed}"
    if not done(run_name):
        print(f"\n=== Running: {run_name} ===")
        train_symmetric_tikh_run(run_name=run_name, out_dir=args.out_dir, device=device,
                                 seed=seed, epochs=150, batch_size=64, z_dim=128,
                                 gamma_tik=1e-6, use_sn=False, save_every=500)

else:
    from run_symmetric_tikh_sn import train_symmetric_tikh_sn_run
    run_name = f"symmetric_Tikh_SN_seed{seed}"
    if not done(run_name):
        print(f"\n=== Running: {run_name} ===")
        train_symmetric_tikh_sn_run(run_name=run_name, out_dir=args.out_dir, device=device,
                                    seed=seed, epochs=150, batch_size=64, z_dim=128,
                                    gamma_tik=1e-6, save_every=500)
