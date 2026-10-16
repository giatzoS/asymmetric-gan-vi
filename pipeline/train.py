import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils
from tqdm import tqdm

from data import build_train_loader
from models import build_discriminator, build_generator, count_parameters
from regularizers import asymmetric_gp_real_fake, tikhonov_params


@dataclass
class RunSpec:
    experiment: str
    dataset: str
    architecture: str
    image_size: int
    data_root: str
    method: str
    seed: int
    max_steps: int
    batch_size: int = 64
    z_dim: int = 128
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.9
    num_workers: int = 2
    split_seed: int = 123
    train_fraction: float = 0.80
    split_protocol: str = "dataset_default"
    save_every: int = 1000
    proxy_save_every: int = 2000
    log_every: int = 100


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_latent_generator(device: str, seed: int):
    # A dedicated latent RNG prevents SN internals from changing z sequences.
    gen_device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=gen_device)
    g.manual_seed(seed)
    return g


def _randn_latent(batch_size: int, z_dim: int, device: str, generator: torch.Generator):
    return torch.randn(batch_size, z_dim, device=device, generator=generator)


def grad_l2_norm_sq(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total


def _finite_or_raise(name: str, value):
    if torch.is_tensor(value):
        ok = bool(torch.isfinite(value.detach()).all().item())
        val = float(value.detach().mean().item())
    else:
        val = float(value)
        ok = math.isfinite(val)
    if not ok:
        raise FloatingPointError(f"Non-finite {name}: {val}")


def gan_train_step(G, D, opt_g, opt_d, real, z, method_cfg: Dict):
    """One alternating D->G update.

    The reported `update_grad_norm` is deliberately named a proxy: it combines
    the D gradient used at the discriminator substep and the G gradient used at
    the subsequent generator substep. It is not claimed to be an exact
    pseudogradient evaluated at a single common iterate.
    """
    # ----- D update -----
    opt_d.zero_grad(set_to_none=True)
    fake = G(z)
    fake_det = fake.detach()

    d_real = D(real)
    d_fake = D(fake_det)
    d_loss = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()

    reg_d = torch.zeros((), device=real.device)
    if method_cfg.get("use_asym_gp", False):
        reg_d = reg_d + asymmetric_gp_real_fake(
            D, real, fake_det, float(method_cfg.get("gamma_gp", 0.0))
        )
    if method_cfg.get("use_tikh_d", False):
        reg_d = reg_d + tikhonov_params(D, float(method_cfg.get("gamma_tik", 0.0)))

    total_d_loss = d_loss + reg_d
    _finite_or_raise("total_d_loss", total_d_loss)
    total_d_loss.backward()
    d_norm_sq = grad_l2_norm_sq(D.parameters())
    opt_d.step()

    # ----- G update -----
    for p in D.parameters():
        p.requires_grad_(False)

    opt_g.zero_grad(set_to_none=True)
    fake_for_g = G(z)
    d_fake_for_g = D(fake_for_g)
    g_loss = F.softplus(-d_fake_for_g).mean()

    reg_g = torch.zeros((), device=real.device)
    if method_cfg.get("use_tikh_g", False):
        reg_g = reg_g + tikhonov_params(G, float(method_cfg.get("gamma_tik", 0.0)))

    total_g_loss = g_loss + reg_g
    _finite_or_raise("total_g_loss", total_g_loss)
    total_g_loss.backward()
    g_norm_sq = grad_l2_norm_sq(G.parameters())
    opt_g.step()

    for p in D.parameters():
        p.requires_grad_(True)

    update_grad_norm = math.sqrt(max(g_norm_sq + d_norm_sq, 0.0))
    _finite_or_raise("update_grad_norm", update_grad_norm)

    return {
        "d_loss": float(d_loss.detach().item()),
        "g_loss": float(g_loss.detach().item()),
        "reg_d": float(reg_d.detach().item()),
        "reg_g": float(reg_g.detach().item()),
        "d_real": float(d_real.detach().mean().item()),
        "d_fake": float(d_fake.detach().mean().item()),
        "update_grad_norm": float(update_grad_norm),
    }


def train_one_run(
    spec: RunSpec,
    method_cfg: Dict,
    device: str,
    output_root: str = "./outputs",
    skip_existing: bool = True,
):
    run_dir = os.path.join(output_root, spec.experiment, f"{spec.method}_seed{spec.seed}")
    os.makedirs(run_dir, exist_ok=True)
    complete_path = os.path.join(run_dir, "training_complete.json")

    if skip_existing and os.path.exists(complete_path):
        print(f"[SKIP] already complete: {run_dir}")
        return run_dir

    seed_everything(spec.seed)

    # Fresh loader per method/run => matched data order for a shared seed.
    loader = build_train_loader(
        dataset_name=spec.dataset,
        data_root=spec.data_root,
        image_size=spec.image_size,
        batch_size=spec.batch_size,
        device=device,
        seed=spec.seed + 30_000,
        num_workers=spec.num_workers,
        split_seed=spec.split_seed,
        train_fraction=spec.train_fraction,
    )

    # Independent model seeds keep initialization reproducible and prevent the
    # presence/absence of SN from perturbing generator initialization.
    torch.manual_seed(spec.seed + 10_000)
    G = build_generator(spec.architecture, spec.image_size, spec.z_dim).to(device)
    torch.manual_seed(spec.seed + 20_000)
    D = build_discriminator(spec.architecture, spec.image_size, method_cfg["use_sn"]).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=spec.lr_g, betas=(spec.beta1, spec.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=spec.lr_d, betas=(spec.beta1, spec.beta2))

    latent_gen = _make_latent_generator(device, spec.seed + 40_000)
    vis_gen = _make_latent_generator(device, spec.seed + 50_000)
    vis_z = _randn_latent(64, spec.z_dim, device, vis_gen)

    cfg_to_save = asdict(spec)
    cfg_to_save["method_config"] = dict(method_cfg)
    cfg_to_save["generator_parameters"] = count_parameters(G)
    cfg_to_save["discriminator_parameters"] = count_parameters(D)
    cfg_to_save["torch_version"] = torch.__version__
    cfg_to_save["device"] = device
    if device.startswith("cuda") and torch.cuda.is_available():
        cfg_to_save["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    proxy_values = []
    data_iter = iter(loader)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="") as log_f:
        writer = csv.DictWriter(
            log_f,
            fieldnames=["step", "d_loss", "g_loss", "reg_d", "reg_g", "d_real", "d_fake", "update_grad_norm"],
        )
        writer.writeheader()

        pbar = tqdm(range(spec.max_steps), desc=f"{spec.experiment}/{spec.method}/seed{spec.seed}")
        for step in pbar:
            try:
                real, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real, _ = next(data_iter)

            real = real.to(device, non_blocking=True)
            z = _randn_latent(real.size(0), spec.z_dim, device, latent_gen)
            stats = gan_train_step(G, D, opt_g, opt_d, real, z, method_cfg)
            proxy_values.append(stats["update_grad_norm"])

            if step % spec.log_every == 0 or step == spec.max_steps - 1:
                writer.writerow({"step": step, **stats})
                log_f.flush()

            if step % spec.save_every == 0 or step == spec.max_steps - 1:
                G.eval()
                with torch.no_grad():
                    samples = G(vis_z)
                    grid = utils.make_grid(samples, normalize=True, value_range=(-1, 1))
                    utils.save_image(grid, os.path.join(run_dir, f"samples_step_{step:07d}.png"))
                G.train()

            if step > 0 and step % spec.proxy_save_every == 0:
                torch.save(torch.tensor(proxy_values), os.path.join(run_dir, "update_grad_norm.pt"))

            if step % 20 == 0:
                pbar.set_postfix(
                    d=f"{stats['d_loss']:.3f}",
                    g=f"{stats['g_loss']:.3f}",
                    reg=f"{stats['reg_d']:.3g}",
                    proxy=f"{stats['update_grad_norm']:.3g}",
                )

    torch.save(G.state_dict(), os.path.join(run_dir, "G_final.pt"))
    torch.save(D.state_dict(), os.path.join(run_dir, "D_final.pt"))
    torch.save(torch.tensor(proxy_values), os.path.join(run_dir, "update_grad_norm.pt"))

    with open(complete_path, "w") as f:
        json.dump({"complete": True, "steps": spec.max_steps}, f, indent=2)

    print(f"[OK] completed: {run_dir}")
    return run_dir
