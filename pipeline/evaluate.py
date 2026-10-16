import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch_fidelity
from torch.serialization import add_safe_globals
from torchvision import utils

from data import build_eval_dataset
from models import build_generator


add_safe_globals([
    np.ndarray,
    np._core.multiarray._reconstruct,
    np.dtype,
])


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_torch_fidelity_cache():
    cache_dir = os.path.expanduser("~/.cache/torch/fidelity_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def generate_fake_images(G, out_dir, num_images, batch_size, z_dim, device, seed):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(seed)
    G.eval()
    saved = 0
    with torch.no_grad():
        while saved < num_images:
            cur = min(batch_size, num_images - saved)
            z = torch.randn(cur, z_dim, device=device)
            fake = G(z)
            for i in range(cur):
                utils.save_image(
                    fake[i],
                    os.path.join(out_dir, f"{saved+i:06d}.png"),
                    normalize=True,
                    value_range=(-1, 1),
                )
            saved += cur


def prepare_reference_dir(cfg, experiment_dir, data_root_override=None, max_images=None):
    dataset = cfg["dataset"]
    if dataset == "cifar10":
        return "cifar10-train", None

    data_root = data_root_override or cfg["data_root"]
    eval_ds = build_eval_dataset(
        dataset_name=dataset,
        data_root=data_root,
        image_size=int(cfg["image_size"]),
        split_seed=int(cfg.get("split_seed", 123)),
        train_fraction=float(cfg.get("train_fraction", 0.90)),
    )

    n = len(eval_ds) if max_images is None else min(len(eval_ds), int(max_images))
    ref_dir = os.path.join(
        experiment_dir,
        "_references",
        f"{dataset}_{cfg['image_size']}px_{cfg.get('split_protocol', 'dataset_default')}_n{n}",
    )

    existing = 0
    if os.path.isdir(ref_dir):
        existing = len([f for f in os.listdir(ref_dir) if f.endswith(".png")])
    if existing == n:
        return ref_dir, n

    if os.path.exists(ref_dir):
        shutil.rmtree(ref_dir)
    os.makedirs(ref_dir, exist_ok=True)
    print(f"[INFO] Exporting {n} held-out real images to {ref_dir}")
    for i in range(n):
        x, _ = eval_ds[i]
        utils.save_image(x, os.path.join(ref_dir, f"{i:06d}.png"))
    return ref_dir, n


def evaluate_run(run_dir, device, requested_num_images=10_000, batch_size=64, data_root_override=None, force=False, with_is=None):
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    out_json = os.path.join(run_dir, "metrics.json")
    if os.path.exists(out_json) and not force:
        print(f"[SKIP] metrics exist: {run_dir}")
        with open(out_json, "r") as f:
            return json.load(f)

    G = build_generator(cfg["architecture"], int(cfg["image_size"]), int(cfg["z_dim"])).to(device)
    G.load_state_dict(torch.load(os.path.join(run_dir, "G_final.pt"), map_location=device))

    experiment_dir = os.path.dirname(run_dir)
    input2, reference_count = prepare_reference_dir(
        cfg, experiment_dir, data_root_override=data_root_override,
        max_images=requested_num_images if cfg["dataset"] != "cifar10" else None,
    )

    num_images = requested_num_images
    if reference_count is not None:
        num_images = min(num_images, reference_count)

    fake_dir = os.path.join(run_dir, f"eval_samples_{num_images}")
    generate_fake_images(
        G, fake_dir, num_images, batch_size, int(cfg["z_dim"]), device,
        seed=int(cfg.get("seed", 0)) + 70_000,
    )

    compute_is = cfg["dataset"] == "cifar10" if with_is is None else bool(with_is)
    print(f"[INFO] Metrics for {os.path.basename(run_dir)} with n={num_images}; IS={compute_is}")

    try:
        raw = torch_fidelity.calculate_metrics(
            input1=fake_dir,
            input2=input2,
            fid=True,
            isc=compute_is,
            kid=True,
            cuda=device.startswith("cuda") and torch.cuda.is_available(),
            verbose=False,
        )
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ["Weights only load failed", "Unsupported global", "UnpicklingError"]):
            print("[WARN] torch-fidelity cache issue; clearing cache and retrying once")
            clear_torch_fidelity_cache()
            raw = torch_fidelity.calculate_metrics(
                input1=fake_dir,
                input2=input2,
                fid=True,
                isc=compute_is,
                kid=True,
                cuda=device.startswith("cuda") and torch.cuda.is_available(),
                verbose=False,
            )
        else:
            raise

    result = {
        **raw,
        "evaluation_num_images": num_images,
        "reference": input2,
        "compute_is": compute_is,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] saved {out_json}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, help="e.g. outputs/plantvillage_dcgan64")
    parser.add_argument("--run-dir", help="Evaluate one run only")
    parser.add_argument("--data-root", help="Override PlantVillage path saved in config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-images", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--with-is", action="store_true", help="Also compute Inception Score (off by default for PlantVillage)")
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [args.run_dir]
    else:
        run_dirs = []
        for name in sorted(os.listdir(args.experiment_dir)):
            path = os.path.join(args.experiment_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "G_final.pt")):
                run_dirs.append(path)

    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "config.json"), "r") as f:
            cfg = json.load(f)
        with_is = args.with_is or cfg["dataset"] == "cifar10"
        evaluate_run(
            run_dir,
            device=args.device,
            requested_num_images=args.num_images,
            batch_size=args.batch_size,
            data_root_override=args.data_root,
            force=args.force,
            with_is=with_is,
        )


if __name__ == "__main__":
    main()
