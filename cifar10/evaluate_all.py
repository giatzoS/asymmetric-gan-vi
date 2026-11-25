# evaluate_all.py
import os
import json
import argparse
import shutil
import traceback

# Import the function from your working evaluate_run.py
from evaluate_run import evaluate_run


def clear_torch_fidelity_cache():
    """
    Torch-fidelity caches CIFAR stats/features here.
    With PyTorch 2.6+ it may crash when loading cached .pt files.
    Safest fix: delete cache and recompute cleanly.
    """
    cache_dir = os.path.expanduser("~/.cache/torch/fidelity_cache")
    if os.path.exists(cache_dir):
        print(f"[CACHE] Removing torch-fidelity cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)


def list_run_dirs(outputs_dir):
    """
    Return list of run directories (subfolders inside outputs_dir).
    """
    run_dirs = []
    for name in sorted(os.listdir(outputs_dir)):
        full = os.path.join(outputs_dir, name)
        if os.path.isdir(full):
            run_dirs.append(full)
    return run_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="./outputs", help="Folder containing run subfolders")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clear_cache_first", action="store_true", help="Delete torch-fidelity cache before evaluating")
    args = parser.parse_args()

    outputs_dir = args.outputs_dir
    if not os.path.exists(outputs_dir):
        raise FileNotFoundError(f"Missing outputs_dir: {outputs_dir}")

    if args.clear_cache_first:
        clear_torch_fidelity_cache()

    run_dirs = list_run_dirs(outputs_dir)
    print(f"\nFound {len(run_dirs)} runs inside: {outputs_dir}\n")

    summary = []
    failed = []

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        g_path = os.path.join(run_dir, "G_final.pt")
        if not os.path.exists(g_path):
            print(f"[SKIP] {run_name}: no G_final.pt")
            continue

        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            print(f"[SKIP] {run_name}: metrics.json already exists")
            continue
    
        print(f"\n=== Evaluating: {run_name} ===")

        # Try once normally
        try:
            metrics = evaluate_run(
                run_dir=run_dir,
                device=args.device,
                z_dim=args.z_dim,
                num_images=args.num_images,
                batch_size=args.batch_size,
            )
            summary.append({"run": run_name, **metrics})
            print(f"[OK] {run_name} done.")
            continue

        except Exception as e:
            msg = str(e)

            # Detect the specific PyTorch 2.6 cache unpickling issue
            retryable = (
                "Weights only load failed" in msg
                or "Unsupported global" in msg
                or "UnpicklingError" in msg
            )

            if retryable:
                print(f"[WARN] {run_name} failed due to cache issue. Clearing cache and retrying once...")
                clear_torch_fidelity_cache()

                try:
                    metrics = evaluate_run(
                        run_dir=run_dir,
                        device=args.device,
                        z_dim=args.z_dim,
                        num_images=args.num_images,
                        batch_size=args.batch_size,
                    )
                    summary.append({"run": run_name, **metrics})
                    print(f"[OK] {run_name} done after cache reset.")
                    continue
                except Exception as e2:
                    print(f"[FAIL] {run_name} crashed again after retry.")
                    print("Reason:", repr(e2))
                    failed.append({"run": run_name, "error": repr(e2)})
                    continue

            # Non-retryable failure
            print(f"[FAIL] {run_name} crashed (non-retryable).")
            print("Reason:", repr(e))
            failed.append({"run": run_name, "error": repr(e)})

    # Save global summary
    summary_path = os.path.join(outputs_dir, "all_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"success": summary, "failed": failed}, f, indent=2)

    print("\n==============================")
    print("✅ Done evaluating all runs.")
    print(f"Saved summary to: {summary_path}")
    print("==============================")

    if len(summary) > 0:
        print("\n✅ Successful runs:")
        for r in summary:
            print(f" - {r['run']} | FID={r.get('frechet_inception_distance', 'NA')}")
    else:
        print("\nNo successful runs.")

    if len(failed) > 0:
        print("\n❌ Failed runs:")
        for r in failed:
            print(f" - {r['run']} | {r['error']}")


if __name__ == "__main__":
    main()

