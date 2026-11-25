#!/usr/bin/env bash
# Extra ESANN runs. Usage (from pipeline/):  bash esann_extra_runs.sh tik|gp
set -uo pipefail
PV_ROOT="${PV_ROOT:-./data/plantvillage}"
SEEDS10=0,1,2,3,4,5,6,7,8,9

case "${1:-}" in
  tik)  # Asym-Tik vs Sym-Tik at gamma=1e-3, CIFAR-10, 10 seeds, same code path
    OUT=./outputs_asym_vs_sym_1e-3
    python run_experiment.py --experiment cifar10_dcgan32 --stage final --max-steps 117300 \
      --seeds $SEEDS10 --methods asym_tik,sym_tik --gamma-tik 1e-3 \
      --data-root /data/P70093283/WCCIproj/A-VI-GAN/data --output-root $OUT --save-every 5000 --num-workers 8
    python evaluate.py --experiment-dir $OUT/cifar10_dcgan32 --data-root /data/P70093283/WCCIproj/A-VI-GAN/data --num-images 10000
    ;;
  gp)   # gamma_GP sweep for ZC-GP + SN, PlantVillage, seeds 0-2 (0.5 already exists)
    for G in 0.05 0.2 2.0; do
      OUT=./outputs_gp_sweep/gamma_$G
      python run_experiment.py --experiment plantvillage_dcgan64 --stage final --max-steps 30000 \
        --seeds 0,1,2 --methods zc_gp_sn --gamma-gp $G \
        --data-root "$PV_ROOT" --output-root $OUT --save-every 3000 --num-workers 8
      python evaluate.py --experiment-dir $OUT/plantvillage_dcgan64 --data-root "$PV_ROOT" --num-images 10000
    done
    ;;
  *) echo "usage: bash esann_extra_runs.sh tik|gp"; exit 1 ;;
esac
python summarize_extra.py
echo "=== DONE $1 ==="
