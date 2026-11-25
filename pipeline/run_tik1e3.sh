#!/bin/bash
# usage: bash run_tik1e3.sh WORKER_ID N_WORKERS GPU
set -u
VENV=/data/P70093283/WCCIproj/A-VI-GAN/venv/bin/activate
PV=/data/P70093283/datasets/plantvillage
CIFAR=/data/P70093283/WCCIproj/A-VI-GAN/data
OUT=./outputs_tik1e3
source $VENV
export CUDA_VISIBLE_DEVICES=$3
JOBS=()
for s in 0 1 2 3 4 5 6 7 8 9; do
  JOBS+=("cifar10_dcgan32 asym_tik_sn 117300 $CIFAR $s 5000")
  JOBS+=("cifar10_dcgan32 sym_tik_sn 117300 $CIFAR $s 5000")
  JOBS+=("plantvillage_dcgan64 asym_tik 30000 $PV $s 3000")
  JOBS+=("plantvillage_dcgan64 asym_tik_sn 30000 $PV $s 3000")
done
for i in "${!JOBS[@]}"; do
  [ $(( i % $2 )) -eq $1 ] || continue
  read EXP M STEPS ROOT S SAVE <<< "${JOBS[$i]}"
  if ls $OUT/$EXP/${M}_seed${S}/training_complete.json >/dev/null 2>&1; then echo "skip $EXP $M $S"; continue; fi
  echo "[$(date +%H:%M)] start $EXP $M seed $S on GPU $3"
  python run_experiment.py --experiment $EXP --stage final --max-steps $STEPS \
    --seeds $S --methods $M --gamma-tik 1e-3 --data-root $ROOT --output-root $OUT \
    --save-every $SAVE --num-workers 2 > logs_tik1e3/${EXP}_${M}_seed${S}.txt 2>&1
  echo "[$(date +%H:%M)] done  $EXP $M seed $S (exit $?)"
done
echo "worker $1 finished"
