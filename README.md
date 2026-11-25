# Asymmetric GAN Training from a Variational Inequality Viewpoint

Code and results for

> S. C. Giagtzoglou, M. H. M. Winands, B. Franci.
> *Asymmetric GAN Training from a Variational Inequality Viewpoint.*
> Submitted to ESANN 2027.

We study discriminator-side (asymmetric) stabilization of GAN training through the lens of
variational inequalities. We compare a zero-centered gradient penalty (ZC-GP) and Tikhonov
regularization, each with and without spectral normalization (SN), on CIFAR-10 (32×32) and
PlantVillage (64×64), and report training dynamics, sample quality (FID, KID, IS) and
computational cost over 10 seeds per configuration.

## Repository layout

```text
cifar10/      CIFAR-10 main study (training, evaluation)
pipeline/     PlantVillage study, runtime profiling and the additional experiments
analysis/     export of per-seed metrics
results/      per-run metrics and statistics
```

Method names in the code: `baseline_plain`, `baseline_sn`, ZC-GP = `zc_gp` (`proposed_GP` in
`cifar10/`), Asym-Tik = `asym_tik` (`proposed_Tikh`), Sym-Tik = `sym_tik` (`symmetric_Tikh`);
the suffix `_sn` / `_SN` denotes spectral normalization.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 is downloaded by torchvision. PlantVillage (RGB images with the official
leaf-group-aware split) is available from
[`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage); pass its
location with `--data-root`.

## Reproducing the results

```bash
# CIFAR-10 main study
cd cifar10
for SEED in 0 1 2 3 4 5 6 7 8 9; do bash run_seed.sh $SEED 0; done
python evaluate_all.py --outputs_dir ./outputs --num_images 10000

# PlantVillage
cd ../pipeline
python run_experiment.py --experiment plantvillage_dcgan64 --stage final --max-steps 30000 \
  --seeds 0,1,2,3,4,5,6,7,8,9 --data-root <plantvillage-root> --output-root ./outputs_final_pv30k
python evaluate.py --experiment-dir outputs_final_pv30k/plantvillage_dcgan64 \
  --data-root <plantvillage-root> --num-images 10000
```

Additional experiments (all in `pipeline/`):

- Tikhonov regularization at γ_Tik = 10⁻³: `esann_extra_runs.sh tik`, `run_tik1e3.sh`
- γ_GP sweep for ZC-GP and ZC-GP + SN: `esann_extra_runs.sh gp`, `wait_eval_gp.sh`
- fresh-seed confirmation on PlantVillage (seeds 10–19): `summarize_confirm.py`
- runtime: `profile_runtime_core.py`
- Fig. 1: `plot_fig1_esann.py`

## Results

`results/all_runs.csv` contains FID, IS and KID for every run. `results/statistics.md` reports
mean ± std over seeds and pairwise Welch t-tests with 95% confidence intervals and Holm
correction; both are produced by `python collect_results.py`. In the tables `cifar10_main` and
`plantvillage_main`, the Tikhonov rows correspond to γ_Tik = 10⁻⁶; the paper reports Tikhonov
at γ_Tik = 10⁻³ (tables `tik1e-3`).

`results/metrics_*.json` contain the per-seed metrics of the main study and of the preliminary
γ_Tik sweep (`metrics_tik_sensitivity.json`), `results/runtime/` the core-update runtimes, and
`results/figures/fig1_dynamics.pdf` is Fig. 1 of the paper.

## License

MIT, see `LICENSE`.