# Asymmetric GAN Training from a Variational Inequality Viewpoint

Code and results for the paper

> S. C. Giagtzoglou, M. H. M. Winands, B. Franci.
> *Asymmetric GAN Training from a Variational Inequality Viewpoint.*
> BNAIC/BeNeLearn 2026.

The paper compares discriminator-side stabilization mechanisms for GAN training
(zero-centered gradient penalty, asymmetric and symmetric Tikhonov regularization,
spectral normalization) under a matched-update protocol, and reports their effect on
training dynamics, sample quality (FID/KID/IS) and computational cost on CIFAR-10
(32x32) and PlantVillage (64x64), with 10 seeds per configuration.

## Repository layout

```text
cifar10/      CIFAR-10 main study: training, evaluation, Table 4 and Fig. 1
pipeline/     PlantVillage study, runtime profiling, Tikhonov sensitivity (Appendix B)
analysis/     export of per-seed metrics and statistical tests
results/      per-seed metrics (JSON), runtime and statistics (CSV), paper figures
```

The two code folders are kept separate because the CIFAR-10 main results were
produced with the code in `cifar10/`, while the PlantVillage study, the runtime
measurements and the Tikhonov sweep were produced with the code in `pipeline/`.
Both use the same DCGAN-style architecture family, losses, optimizer settings and
regularization coefficients.

Method names used in the code:

| Paper | `cifar10/` | `pipeline/` |
|---|---|---|
| Baseline (plain) | `baseline_plain` | `baseline_plain` |
| Baseline + SN | `baseline_sn` | `baseline_sn` |
| ZC-GP | `proposed_GP` | `zc_gp` |
| ZC-GP + SN | `proposed_GP_SN` | `zc_gp_sn` |
| Asym-Tik | `proposed_Tikh` | `asym_tik` |
| Asym-Tik + SN | `proposed_Tikh_SN` | `asym_tik_sn` |
| Sym-Tik | `symmetric_Tikh` | `sym_tik` |
| Sym-Tik + SN | `symmetric_Tikh_SN` | `sym_tik_sn` |

## Installation

Experiments were run with Python 3.9 and PyTorch 2.6.0 (CUDA 12.4) on an NVIDIA RTX A5000.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

- **CIFAR-10** is downloaded automatically by torchvision into `./data`.
- **PlantVillage** (RGB `color` images, official leaf-group-aware train/test split) is
  expected in the following layout:

  ```text
  <plantvillage-root>/
  ├── extracted/                 image files
  └── splits/
      ├── color_train.txt        relative image paths, one per line (43,596 images)
      └── color_test.txt         relative image paths, one per line (10,709 images)
  ```

  Both the images and the split files come from the Hugging Face dataset
  [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage):

  ```bash
  mkdir -p data/plantvillage/splits && cd data/plantvillage
  wget https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip
  wget -P splits https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_train.txt
  wget -P splits https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_test.txt
  unzip -q data.zip -d extracted      # creates extracted/raw/color/<class>/<image>
  ```

  Pass `<plantvillage-root>` with `--data-root`.

## Reproducing the paper

The per-seed metrics behind every table are in `results/`, so the tables and the
statistical tests can be checked without retraining (see *Statistical tests* below).

### CIFAR-10 (Table 4, Fig. 1)

```bash
cd cifar10
for SEED in 0 1 2 3 4 5 6 7 8 9; do bash run_seed.sh $SEED 0; done   # 8 configurations per seed
python evaluate_all.py --outputs_dir ./outputs --num_images 10000 --batch_size 64
python make_results_table.py                                         # FID / IS / KID, mean ± std
python plot_dynamics.py --out_dir ./outputs --ema_alpha 0.02         # Fig. 1
```

Seeds 0–4 were originally produced with `run_all.py`, `run_symmetric_tikh.py` and
`run_symmetric_tikh_sn.py`; seeds 5–9 with `run_more_seeds.py`, which calls the same
training functions with the same settings. Each seed is independent, so seeds can be
run in parallel.

### PlantVillage (Table 5, Fig. 2)

```bash
cd pipeline
python run_experiment.py --experiment plantvillage_dcgan64 --stage final \
  --max-steps 30000 --seeds 0,1,2,3,4,5,6,7,8,9 \
  --methods baseline_plain,baseline_sn,zc_gp,zc_gp_sn,asym_tik,asym_tik_sn \
  --data-root <plantvillage-root> --output-root ./outputs_final_pv30k --save-every 3000
python evaluate.py --experiment-dir outputs_final_pv30k/plantvillage_dcgan64 \
  --data-root <plantvillage-root> --num-images 10000
python summarize_pv.py              # FID / KID, mean ± std
python plot_pv_paper.py             # Fig. 2 (written to figures/)
python summarize_pv_dynamics.py     # dynamics summary
```

### Runtime (runtime columns of Tables 4 and 5)

```bash
cd pipeline
python profile_runtime_core.py --experiment cifar10_dcgan32 \
  --methods baseline_plain,baseline_sn,zc_gp,zc_gp_sn,asym_tik,asym_tik_sn,sym_tik,sym_tik_sn \
  --warmup 50 --measure 200 --seed 0 --output-root ./outputs_runtime_cifar_final
python profile_runtime_core.py --experiment plantvillage_dcgan64 \
  --data-root <plantvillage-root> --warmup 50 --measure 200 --seed 0 \
  --output-root ./outputs_runtime_core
```

The core-update time covers one discriminator and one generator update, including
regularization and SN, after warm-up and with explicit GPU synchronization.

### Tikhonov sensitivity (Appendix B)

```bash
cd pipeline
for G in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1; do
  python run_experiment.py --experiment cifar10_dcgan32 --stage final --max-steps 117300 \
    --seeds 0,1,2 --methods asym_tik --gamma-tik $G \
    --output-root ./outputs_tik_sensitivity/gamma_$G --save-every 5000
  python evaluate.py --experiment-dir outputs_tik_sensitivity/gamma_$G/cifar10_dcgan32 \
    --data-root ./data --num-images 10000
done
python summarize_tik_sweep.py
```

### PlantVillage pilot (Appendix A)

Single diagnostic run with seed 999, 10,000 steps, evaluated on 2,000 images; not part
of any reported statistic.

```bash
cd pipeline
python run_experiment.py --experiment plantvillage_dcgan64 --stage pilot \
  --data-root <plantvillage-root> --output-root ./outputs_pilot
python evaluate.py --experiment-dir outputs_pilot/plantvillage_dcgan64 \
  --data-root <plantvillage-root> --num-images 2000
```

### Statistical tests

Pairwise comparisons use a two-sided Welch t-test with Holm correction over the 13
comparisons; Mann–Whitney U and a paired t-test over matched seeds are reported as
robustness checks.

```bash
python analysis/stats_tests.py                    # FID, from results/*.json
python analysis/stats_tests.py --metric kid       # KID
```

After retraining, `--source runs` reads the per-run `metrics.json` files instead, and
`analysis/export_metrics.py` regenerates the JSON files in `results/`.

## Results folder

| File | Content |
|---|---|
| `metrics_cifar10.json` | per-seed FID, KID, IS for the 8 CIFAR-10 configurations, seeds 0–9 |
| `metrics_plantvillage.json` | per-seed FID, KID for the 6 PlantVillage configurations, seeds 0–9 |
| `metrics_tik_sensitivity.json` | per-seed metrics of the Asym-Tik coefficient sweep, seeds 0–2 |
| `runtime/` | core-update runtime per configuration (CIFAR-10, PlantVillage) |
| `stats/` | output of `analysis/stats_tests.py` for FID and KID |
| `figures/` | training-dynamics figures (Figs. 1–2) and the FID summary chart |

## License

MIT, see `LICENSE`.
