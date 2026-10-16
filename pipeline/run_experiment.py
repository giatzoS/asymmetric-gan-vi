import argparse
import copy
import os

import torch

from configs import DEFAULT_METHODS, EXPERIMENTS, METHOD_CONFIGS, STAGES
from train import RunSpec, train_one_run


def parse_csv(value, cast=str):
    if value is None:
        return None
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run controlled GAN stabilization experiments")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), default="smoke")
    parser.add_argument("--data-root", default="./data", help="CIFAR root or Hugging Face cache directory for PlantVillage")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--methods", help="Comma-separated override, e.g. baseline_plain,zc_gp")
    parser.add_argument("--seeds", help="Comma-separated integer override")
    parser.add_argument("--max-steps", type=int, help="Override step budget; REQUIRED for final stage")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gamma-gp", type=float, help="Override GP coefficient for GP methods")
    parser.add_argument("--gamma-tik", type=float, help="Override Tikhonov coefficient for Tikh methods")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    exp = copy.deepcopy(EXPERIMENTS[args.experiment])
    stage = STAGES[args.stage]

    methods = parse_csv(args.methods) or list(stage["methods"])
    seeds = parse_csv(args.seeds, int) or list(stage["seeds"])
    max_steps = args.max_steps if args.max_steps is not None else stage["max_steps"]

    if args.stage == "final" and max_steps is None:
        raise SystemExit(
            "Final stage requires --max-steps. Choose and freeze the full budget after inspecting the pilot."
        )
    if max_steps is None or max_steps <= 0:
        raise SystemExit("max_steps must be a positive integer")

    unknown = [m for m in methods if m not in METHOD_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Available: {sorted(METHOD_CONFIGS)}")

    batch_size = args.batch_size or exp["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== Experiment plan ===")
    print(f"experiment : {args.experiment}")
    print(f"stage      : {args.stage}")
    print(f"dataset    : {exp['dataset']}")
    print(f"architecture: {exp['architecture']} @ {exp['image_size']}x{exp['image_size']}")
    print(f"methods    : {methods}")
    print(f"seeds      : {seeds}")
    print(f"max_steps  : {max_steps}")
    print(f"batch_size : {batch_size}")
    print(f"device     : {device}\n")

    if args.stage in {"smoke", "pilot"}:
        print("NOTE: smoke/pilot results are diagnostics only; seed 999 is excluded from final paper results.\n")

    for seed in seeds:
        for method in methods:
            mcfg = copy.deepcopy(METHOD_CONFIGS[method])
            if args.gamma_gp is not None and mcfg.get("use_asym_gp", False):
                mcfg["gamma_gp"] = args.gamma_gp
            if args.gamma_tik is not None and (mcfg.get("use_tikh_d", False) or mcfg.get("use_tikh_g", False)):
                mcfg["gamma_tik"] = args.gamma_tik

            spec = RunSpec(
                experiment=args.experiment,
                dataset=exp["dataset"],
                architecture=exp["architecture"],
                image_size=exp["image_size"],
                data_root=os.path.abspath(args.data_root),
                method=method,
                seed=seed,
                max_steps=max_steps,
                batch_size=batch_size,
                z_dim=exp["z_dim"],
                lr_g=exp["lr_g"],
                lr_d=exp["lr_d"],
                beta1=exp["betas"][0],
                beta2=exp["betas"][1],
                num_workers=args.num_workers,
                split_seed=exp["split_seed"],
                train_fraction=exp["train_fraction"],
                split_protocol=exp.get("split_protocol", "dataset_default"),
                save_every=args.save_every,
            )
            train_one_run(
                spec=spec,
                method_cfg=mcfg,
                device=device,
                output_root=args.output_root,
                skip_existing=not args.no_skip_existing,
            )


if __name__ == "__main__":
    main()
