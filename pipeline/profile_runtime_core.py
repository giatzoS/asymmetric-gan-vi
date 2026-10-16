import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from configs import DEFAULT_METHODS, EXPERIMENTS, METHOD_CONFIGS
from data import build_train_loader
from models import build_discriminator, build_generator
from regularizers import asymmetric_gp_real_fake, tikhonov_params
from train import _make_latent_generator, _randn_latent


def seed_for_profiling(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Match the original runtime profiler: realistic throughput, not
    # deterministic-training settings.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def sync(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def core_train_step(G, D, opt_g, opt_d, real, z, cfg):
    # ----- D update -----
    opt_d.zero_grad(set_to_none=True)

    fake = G(z)
    fake_det = fake.detach()

    d_real = D(real)
    d_fake = D(fake_det)
    d_loss = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()

    reg_d = torch.zeros((), device=real.device)

    if cfg.get("use_asym_gp", False):
        reg_d = reg_d + asymmetric_gp_real_fake(
            D,
            real,
            fake_det,
            float(cfg.get("gamma_gp", 0.0)),
        )

    if cfg.get("use_tikh_d", False):
        reg_d = reg_d + tikhonov_params(
            D,
            float(cfg.get("gamma_tik", 0.0)),
        )

    (d_loss + reg_d).backward()
    opt_d.step()

    # ----- G update -----
    for p in D.parameters():
        p.requires_grad_(False)

    opt_g.zero_grad(set_to_none=True)

    fake_for_g = G(z)
    d_fake_for_g = D(fake_for_g)
    g_loss = F.softplus(-d_fake_for_g).mean()

    reg_g = torch.zeros((), device=real.device)

    if cfg.get("use_tikh_g", False):
        reg_g = reg_g + tikhonov_params(
            G,
            float(cfg.get("gamma_tik", 0.0)),
        )

    (g_loss + reg_g).backward()
    opt_g.step()

    for p in D.parameters():
        p.requires_grad_(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measure", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows = []
    print("\n=== Core-update runtime profiling ===")

    for method in methods:
        seed_for_profiling(args.seed)
        cfg = dict(METHOD_CONFIGS[method])

        loader = build_train_loader(
            exp["dataset"],
            args.data_root,
            exp["image_size"],
            exp["batch_size"],
            device,
            seed=args.seed + 30_000,
            num_workers=2,
            split_seed=exp["split_seed"],
            train_fraction=exp["train_fraction"],
        )
        data_it = iter(loader)

        torch.manual_seed(args.seed + 10_000)
        G = build_generator(
            exp["architecture"],
            exp["image_size"],
            exp["z_dim"],
        ).to(device)

        torch.manual_seed(args.seed + 20_000)
        D = build_discriminator(
            exp["architecture"],
            exp["image_size"],
            cfg["use_sn"],
        ).to(device)

        opt_g = torch.optim.Adam(
            G.parameters(),
            lr=exp["lr_g"],
            betas=exp["betas"],
        )
        opt_d = torch.optim.Adam(
            D.parameters(),
            lr=exp["lr_d"],
            betas=exp["betas"],
        )

        zg = _make_latent_generator(device, args.seed + 40_000)

        times = []
        total = args.warmup + args.measure

        for k in range(total):
            try:
                real, _ = next(data_it)
            except StopIteration:
                data_it = iter(loader)
                real, _ = next(data_it)

            real = real.to(device, non_blocking=True)
            z = _randn_latent(
                real.size(0),
                exp["z_dim"],
                device,
                zg,
            )

            if k == args.warmup:
                sync(device)

            t0 = time.perf_counter()

            core_train_step(
                G,
                D,
                opt_g,
                opt_d,
                real,
                z,
                cfg,
            )

            sync(device)
            t1 = time.perf_counter()

            if k >= args.warmup:
                times.append(t1 - t0)

        arr = np.asarray(times, dtype=np.float64)

        row = {
            "method": method,
            "seconds_per_iteration_mean": float(arr.mean()),
            "seconds_per_iteration_std": float(arr.std(ddof=1)),
        }
        rows.append(row)

        print(
            f"{method:18s} "
            f"{row['seconds_per_iteration_mean']:.5f} "
            f"± {row['seconds_per_iteration_std']:.5f} s/iter"
        )

    baseline = next(
        r["seconds_per_iteration_mean"]
        for r in rows
        if r["method"] == "baseline_plain"
    )

    for r in rows:
        r["relative_overhead"] = r["seconds_per_iteration_mean"] / baseline

    out_dir = os.path.join(args.output_root, args.experiment)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "runtime_profile_core.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] saved {out_path}")
    print("\n=== Relative overhead ===")
    for r in rows:
        print(f"{r['method']:18s} {r['relative_overhead']:.2f}x")


if __name__ == "__main__":
    main()
