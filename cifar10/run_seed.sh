#!/bin/bash
# Train all eight CIFAR-10 configurations for one seed.
# usage: bash run_seed.sh SEED [GPU]
set -e
cd "$(dirname "$0")"
mkdir -p logs
export CUDA_VISIBLE_DEVICES=${2:-0}
for G in main sym sym_sn; do
  python run_more_seeds.py --seed "$1" --group "$G" >> "logs/cifar_seed$1.log" 2>&1
done
echo "SEED $1 DONE" >> "logs/cifar_seed$1.log"
