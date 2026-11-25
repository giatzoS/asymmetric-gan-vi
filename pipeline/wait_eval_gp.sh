#!/usr/bin/env bash
# Waits for the 39 sweep runs, evaluates them, prints the summary (run from pipeline/).
PV=/data/P70093283/datasets/plantvillage
C10=/data/P70093283/WCCIproj/A-VI-GAN/data
cnt() { ls outputs_gp_sweep/gamma_{0.05,2.0,10.0}/plantvillage_dcgan64/zc_gp_seed*/training_complete.json \
           outputs_gp_sweep/gamma_{5.0,10.0}/plantvillage_dcgan64/zc_gp_sn_seed*/training_complete.json \
           outputs_gp_sweep_c10/gamma_*/cifar10_dcgan32/*/training_complete.json 2>/dev/null | wc -l; }
while [ "$(cnt)" -lt 39 ]; do echo "$(date +%a\ %H:%M)  finished $(cnt)/39"; sleep 600; done
for G in 0.05 2.0 5.0 10.0; do python evaluate.py --experiment-dir outputs_gp_sweep/gamma_$G/plantvillage_dcgan64 --data-root $PV --num-images 10000; done
for G in 0.05 0.5 2.0 10.0; do python evaluate.py --experiment-dir outputs_gp_sweep_c10/gamma_$G/cifar10_dcgan32 --data-root $C10 --num-images 10000; done
python summarize_gp_full.py | tee summary_gp_full.txt
