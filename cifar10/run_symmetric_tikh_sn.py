# run_symmetric_tikh_sn.py
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils
from tqdm import tqdm

from models import Generator, Discriminator
from data import get_cifar10_loader


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def grad_l2_norm_sq(params):
    """Sum of squared L2 norms of gradients in a parameter list."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total


def tikhonov_params(model, gamma: float):
    """(gamma/2) * ||params||^2"""
    if gamma <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    s = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        s = s + p.pow(2).sum()
    return 0.5 * gamma * s


# -------------------------
# Train ONE run (one seed)
# -------------------------
def train_symmetric_tikh_sn_run(
    run_name: str,
    out_dir: str,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    z_dim: int = 128,
    lr: float = 2e-4,
    betas=(0.0, 0.9),
    gamma_tik: float = 1e-6,
    save_every: int = 500,
):
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    seed_everything(seed)

    # Save config
    cfg = {
        "case": "symmetric_tikhonov",
        "use_sn": True,
        "gamma_tik": gamma_tik,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "z_dim": z_dim,
        "lr": lr,
        "betas": list(betas),
        "save_every": save_every,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Data loader
    loader = get_cifar10_loader(batch_size=batch_size, device=device, seed=seed)

    # Models
    G = Generator(z_dim=z_dim).to(device)
    # SN enabled only for D
    D = Discriminator(use_sn=True).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    vis_z = torch.randn(64, z_dim, device=device)

    gaps = []
    step = 0

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"{run_name} | Epoch {epoch+1}/{epochs}")
        for real, _ in pbar:
            real = real.to(device)
            bs = real.size(0)
            z = torch.randn(bs, z_dim, device=device)

            # -----------------
            # (1) D update
            # -----------------
            opt_d.zero_grad(set_to_none=True)

            fake = G(z)
            fake_det = fake.detach()

            d_real = D(real)
            d_fake = D(fake_det)

            # Non-saturating logistic GAN loss (same style as your other scripts)
            d_loss = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()

            # symmetric Tik: add to D
            reg_d = tikhonov_params(D, gamma_tik)

            total_d_loss = d_loss + reg_d
            total_d_loss.backward()

            d_norm_sq = grad_l2_norm_sq(D.parameters())
            opt_d.step()

            # -----------------
            # (2) G update
            # -----------------
            # Freeze D for G step
            for p in D.parameters():
                p.requires_grad_(False)

            opt_g.zero_grad(set_to_none=True)

            fake_for_g = G(z)
            d_fake_for_g = D(fake_for_g)
            g_loss = F.softplus(-d_fake_for_g).mean()

            # symmetric Tik: add to G
            reg_g = tikhonov_params(G, gamma_tik)

            total_g_loss = g_loss + reg_g
            total_g_loss.backward()

            g_norm_sq = grad_l2_norm_sq(G.parameters())
            opt_g.step()

            # Unfreeze D
            for p in D.parameters():
                p.requires_grad_(True)

            # -----------------
            # (3) Gap proxy
            # -----------------
            gap = (g_norm_sq + d_norm_sq) ** 0.5
            gaps.append(gap)

            pbar.set_postfix({
                "d_real": float(d_real.mean().item()),
                "d_fake": float(d_fake.mean().item()),
                "d_loss": float(d_loss.item()),
                "g_loss": float(g_loss.item()),
                "reg_d": float(reg_d.item()),
                "reg_g": float(reg_g.item()),
                "gap": float(gap),
                "step": step,
            })

            # -----------------
            # (4) Save samples
            # -----------------
            if step % save_every == 0:
                G.eval()
                with torch.no_grad():
                    samples = G(vis_z)
                    grid = utils.make_grid(samples, normalize=True, value_range=(-1, 1))
                    utils.save_image(grid, os.path.join(run_dir, f"step_{step:06d}.png"))
                G.train()

            step += 1

    # Save outputs
    torch.save(G.state_dict(), os.path.join(run_dir, "G_final.pt"))
    torch.save(D.state_dict(), os.path.join(run_dir, "D_final.pt"))
    torch.save(torch.tensor(gaps), os.path.join(run_dir, "gaps.pt"))

    return run_dir


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    seeds = [0,1,2,3,4]
    epochs = 150
    batch_size = 64
    z_dim = 128
    gamma_tik = 1e-6

    for seed in seeds:
        run_name = f"symmetric_Tikh_SN_seed{seed}"
        print(f"\n=== Running: {run_name} ===")
        train_symmetric_tikh_sn_run(
            run_name=run_name,
            out_dir=out_dir,
            device=device,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            z_dim=z_dim,
            gamma_tik=gamma_tik,
            save_every=500,
        )


if __name__ == "__main__":
    main()
