# evaluate_run.py
import os
import json
import shutil
from torchvision import utils
import torch_fidelity
import numpy as np
import torch
import argparse
import random
from torch.serialization import add_safe_globals

from models import Generator

# Allowlist numpy objects inside torch-fidelity cache files (PyTorch 2.6+)
add_safe_globals([
    np.ndarray,
    np._core.multiarray._reconstruct,
    np.dtype,
])

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_fake_images(G, out_dir, num_images, batch_size, z_dim, device):
    os.makedirs(out_dir, exist_ok=True)
    G.eval()
    saved = 0
    with torch.no_grad():
        while saved < num_images:
            cur_bs = min(batch_size, num_images - saved)
            z = torch.randn(cur_bs, z_dim, device=device)
            fake = G(z)  # in [-1, 1] due to Tanh

            for i in range(cur_bs):
                path = os.path.join(out_dir, f"{saved+i:06d}.png")
                utils.save_image(fake[i], path, normalize=True, value_range=(-1, 1))

            saved += cur_bs

def evaluate_run(run_dir, device="cuda", z_dim=128, num_images=10000, batch_size=64, force=False):
    print(f"[INFO] Evaluating run: {run_dir}")
    print(f"[INFO] num_images = {num_images}")

    # ---- load config if present (paper-safe) ----
    cfg_path = os.path.join(run_dir, "config.json")
    seed = 0
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        z_dim = int(cfg.get("z_dim", z_dim))
        seed = int(cfg.get("seed", 0))

    # Reproducible sampling for this evaluation run
    seed_everything(seed)

    # Skip if metrics exist already
    out_json = os.path.join(run_dir, "metrics.json")
    if (not force) and os.path.exists(out_json):
        print(f"[SKIP] metrics.json already exists: {out_json}")
        with open(out_json, "r") as f:
            return json.load(f)

    g_path = os.path.join(run_dir, "G_final.pt")
    if not os.path.exists(g_path):
        raise FileNotFoundError(f"Missing generator checkpoint: {g_path}")

    # Load Generator
    G = Generator(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(g_path, map_location=device))
    G.eval()

    # Create evaluation folder (fresh)
    fake_dir = os.path.join(run_dir, f"eval_samples_{num_images}")
    if os.path.exists(fake_dir):
        shutil.rmtree(fake_dir)

    print(f"[INFO] Generating fake images into: {fake_dir}")
    generate_fake_images(
        G=G,
        out_dir=fake_dir,
        num_images=num_images,
        batch_size=batch_size,
        z_dim=z_dim,
        device=device
    )

    # Sanity check
    num_files = len([f for f in os.listdir(fake_dir) if f.endswith(".png")])
    print(f"[INFO] Found {num_files} PNG files in {fake_dir}")
    if num_files != num_images:
        print(f"[WARN] Expected {num_images} images but found {num_files}")

    print("[INFO] Running torch-fidelity metrics...")
    metrics_raw = torch_fidelity.calculate_metrics(
        input1=fake_dir,          # IS computed here
        input2="cifar10-train",   # real reference for FID/KID
        fid=True,
        isc=True,
        kid=True,
        cuda=(device.startswith("cuda") and torch.cuda.is_available()),
        verbose=False,
    )

    with open(out_json, "w") as f:
        json.dump(metrics_raw, f, indent=2)

    print(f"[OK] Saved metrics to: {out_json}")
    return metrics_raw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Recompute even if metrics.json exists")
    args = parser.parse_args()

    evaluate_run(
        run_dir=args.run_dir,
        device=args.device,
        z_dim=args.z_dim,
        num_images=args.num_images,
        batch_size=args.batch_size,
        force=args.force,
    )

if __name__ == "__main__":
    main()
