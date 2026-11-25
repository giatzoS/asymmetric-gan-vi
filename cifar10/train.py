# train.py
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils
from tqdm import tqdm

from models import Generator, Discriminator
from regularizers import asymmetric_gp_real_fake, tikhonov_params

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for determinism (CUDA)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional but recommended: will raise if something nondeterministic is used
    # torch.use_deterministic_algorithms(True)




def grad_l2_norm_sq(params):
    """Sum of squared L2 norms of gradients in a parameter list."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total


def train_one_case(
    case_name,
    cfg,
    loader,
    device,
    out_dir,
    z_dim=128,
    epochs=50,
    save_every=500,
    seed=0,
):
    run_dir = os.path.join(out_dir, case_name)
    os.makedirs(run_dir, exist_ok=True)

    seed_everything(seed)

    # Save config for paper reproducibility
    cfg_to_save = dict(cfg)
    cfg_to_save["seed"] = seed
    cfg_to_save["epochs"] = epochs
    cfg_to_save["z_dim"] = z_dim

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator(use_sn=cfg["use_sn"]).to(device)

    # If "Tikhonov only" means weight decay, you can do it here instead of loss.
    # But since your paper uses explicit (gamma/2)||D||^2, we keep it as loss term.
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))

    vis_z = torch.randn(64, z_dim, device=device)

    gaps = []
    step = 0

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"{case_name} | Epoch {epoch+1}/{epochs}")
        for real, _ in pbar:
            real = real.to(device)
            bs = real.size(0)
            z = torch.randn(bs, z_dim, device=device)

            # -----------------
            # (1) D update
            # -----------------
            opt_d.zero_grad()
            fake = G(z)
            fake_det = fake.detach()

            d_real = D(real)
            d_fake = D(fake_det)

            d_loss = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()

            reg = torch.tensor(0.0, device=device)

            if cfg["use_asym_gp"]:
                reg = reg + asymmetric_gp_real_fake(D, real, fake_det, cfg["gamma_gp"])

            if cfg["use_tikhonov"]:
                reg = reg + tikhonov_params(D, cfg["gamma_tik"])

            total_d_loss = d_loss + reg
            total_d_loss.backward()

            # ---- proxy part 1: discriminator gradient norm (before step) ----
            d_norm_sq = grad_l2_norm_sq(D.parameters())

            opt_d.step()

            # -----------------
            # (2) G update
            # -----------------
            for p in D.parameters():
                p.requires_grad_(False)

            opt_g.zero_grad()

            fake_for_g = G(z)                  # recompute
            d_fake_for_g = D(fake_for_g)       # D forward (params frozen)
            g_loss = F.softplus(-d_fake_for_g).mean()
            g_loss.backward()

            # ---- proxy part 2: generator gradient norm (before step) ----
            g_norm_sq = grad_l2_norm_sq(G.parameters())

            opt_g.step()
            for p in D.parameters():
                p.requires_grad_(True)

            # -----------------
            # (3) Stationarity proxy (your "gap")
            # -----------------
            with torch.no_grad():
                gap = (g_norm_sq + d_norm_sq) ** 0.5
                gaps.append(gap)

            if step % 2000 == 0 and step > 0:
                torch.save(torch.tensor(gaps), os.path.join(run_dir, "gaps.pt"))

            pbar.set_postfix({
                "d_real": float(d_real.mean().item()),
                "d_fake": float(d_fake.mean().item()),
                "d_loss": float(d_loss.item()),
                "g_loss": float(g_loss.item()),
                "reg": float(reg.item()),
                "gap": float(gap),
                "step": step
            })

            # -----------------
            # (4) Save samples
            # -----------------
            if step % save_every == 0:
                G.eval()
                with torch.no_grad():
                    samples = G(vis_z)   # ✅ use fixed noise
                    grid = utils.make_grid(samples, normalize=True, value_range=(-1, 1))
                    utils.save_image(grid, os.path.join(run_dir, f"step_{step:06d}.png"))
                G.train()

            step += 1

    # Save final models
    torch.save(G.state_dict(), os.path.join(run_dir, "G_final.pt"))
    torch.save(D.state_dict(), os.path.join(run_dir, "D_final.pt"))

    # Save gaps for plotting
    torch.save(torch.tensor(gaps), os.path.join(run_dir, "gaps.pt"))
    return gaps