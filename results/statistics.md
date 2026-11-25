# Statistics

Mean ± std (ddof=1) over seeds; Welch two-sided t-tests, 95% CI of A − B, Holm correction within each table.

## cifar10_main / cifar10_dcgan32

| method | n | FID | IS | KID |
|---|---|---|---|---|
| baseline_plain | 10 | 92.26 ± 6.9 | 4.001 ± 0.26 | 0.08271 ± 0.0094 |
| baseline_sn | 10 | 44.17 ± 1.5 | 6.087 ± 0.19 | 0.02982 ± 0.0013 |
| proposed_GP | 10 | 37.71 ± 0.87 | 6.371 ± 0.088 | 0.02503 ± 0.00072 |
| proposed_GP_SN | 10 | 41.63 ± 1.5 | 6.309 ± 0.14 | 0.02775 ± 0.0011 |
| proposed_Tikh | 10 | 92.35 ± 5.4 | 3.964 ± 0.18 | 0.08182 ± 0.0064 |
| proposed_Tikh_SN | 10 | 44.13 ± 1.4 | 6.003 ± 0.13 | 0.02996 ± 0.0013 |
| symmetric_Tikh | 10 | 93.54 ± 6.9 | 3.932 ± 0.22 | 0.08362 ± 0.0082 |
| symmetric_Tikh_SN | 10 | 43.66 ± 1.8 | 6.041 ± 0.16 | 0.03014 ± 0.0013 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| baseline_plain | baseline_sn | +48.09 | [43.1, 53.07] | 1.2e-09 | 1.9e-08 |
| baseline_plain | proposed_GP | +54.55 | [49.6, 59.5] | 8.3e-10 | 1.5e-08 |
| baseline_plain | proposed_GP_SN | +50.63 | [45.64, 55.61] | 7.5e-10 | 1.5e-08 |
| baseline_plain | proposed_Tikh | -0.08639 | [-5.929, 5.757] | 0.98 | 1 |
| baseline_plain | proposed_Tikh_SN | +48.13 | [43.16, 53.11] | 1.6e-09 | 2.2e-08 |
| baseline_plain | symmetric_Tikh | -1.278 | [-7.741, 5.185] | 0.68 | 1 |
| baseline_plain | symmetric_Tikh_SN | +48.6 | [43.59, 53.61] | 7.5e-10 | 1.5e-08 |
| baseline_sn | proposed_GP | +6.465 | [5.271, 7.658] | 1.2e-08 | 1.5e-07 |
| baseline_sn | proposed_GP_SN | +2.54 | [1.101, 3.98] | 0.0016 | 0.013 |
| baseline_sn | proposed_Tikh | -48.17 | [-52.11, -44.24] | 5e-11 | 1.3e-09 |
| baseline_sn | proposed_Tikh_SN | +0.04656 | [-1.313, 1.406] | 0.94 | 1 |
| baseline_sn | symmetric_Tikh | -49.37 | [-54.33, -44.4] | 9e-10 | 1.5e-08 |
| baseline_sn | symmetric_Tikh_SN | +0.5109 | [-1.06, 2.082] | 0.5 | 1 |
| proposed_GP | proposed_GP_SN | -3.924 | [-5.116, -2.733] | 5.2e-06 | 5.2e-05 |
| proposed_GP | proposed_Tikh | -54.64 | [-58.52, -50.76] | 6.5e-11 | 1.6e-09 |
| proposed_GP | proposed_Tikh_SN | -6.418 | [-7.498, -5.338] | 1.6e-09 | 2.2e-08 |
| proposed_GP | symmetric_Tikh | -55.83 | [-60.76, -50.91] | 6.4e-10 | 1.3e-08 |
| proposed_GP | symmetric_Tikh_SN | -5.954 | [-7.315, -4.592] | 3.5e-07 | 3.8e-06 |
| proposed_GP_SN | proposed_Tikh | -50.72 | [-54.65, -46.78] | 3e-11 | 8.1e-10 |
| proposed_GP_SN | proposed_Tikh_SN | -2.494 | [-3.851, -1.136] | 0.0012 | 0.01 |
| proposed_GP_SN | symmetric_Tikh | -51.91 | [-56.87, -46.95] | 5.6e-10 | 1.2e-08 |
| proposed_GP_SN | symmetric_Tikh_SN | -2.029 | [-3.599, -0.4603] | 0.014 | 0.099 |
| proposed_Tikh | proposed_Tikh_SN | +48.22 | [44.31, 52.14] | 7.9e-11 | 1.9e-09 |
| proposed_Tikh | symmetric_Tikh | -1.192 | [-7.016, 4.633] | 0.67 | 1 |
| proposed_Tikh | symmetric_Tikh_SN | +48.69 | [44.72, 52.65] | 2.2e-11 | 6.1e-10 |
| proposed_Tikh_SN | symmetric_Tikh | -49.41 | [-54.36, -44.46] | 1.2e-09 | 1.8e-08 |
| proposed_Tikh_SN | symmetric_Tikh_SN | +0.4644 | [-1.036, 1.964] | 0.52 | 1 |
| symmetric_Tikh | symmetric_Tikh_SN | +49.88 | [44.89, 54.86] | 5.4e-10 | 1.2e-08 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| baseline_plain | baseline_sn | +0.05289 | [0.04614, 0.05963] | 1.7e-08 | 2.7e-07 |
| baseline_plain | proposed_GP | +0.05768 | [0.05095, 0.0644] | 1e-08 | 1.9e-07 |
| baseline_plain | proposed_GP_SN | +0.05496 | [0.04823, 0.0617] | 1.4e-08 | 2.3e-07 |
| baseline_plain | proposed_Tikh | +0.0008875 | [-0.006732, 0.008507] | 0.81 | 1 |
| baseline_plain | proposed_Tikh_SN | +0.05275 | [0.04601, 0.05949] | 1.8e-08 | 2.7e-07 |
| baseline_plain | symmetric_Tikh | -0.0009061 | [-0.009181, 0.007369] | 0.82 | 1 |
| baseline_plain | symmetric_Tikh_SN | +0.05257 | [0.04583, 0.05932] | 1.8e-08 | 2.7e-07 |
| baseline_sn | proposed_GP | +0.004791 | [0.003753, 0.005828] | 1.2e-07 | 1.3e-06 |
| baseline_sn | proposed_GP_SN | +0.002076 | [0.0009175, 0.003235] | 0.0015 | 0.01 |
| baseline_sn | proposed_Tikh | -0.052 | [-0.05661, -0.04739] | 3.1e-10 | 8.7e-09 |
| baseline_sn | proposed_Tikh_SN | -0.0001375 | [-0.001372, 0.001097] | 0.82 | 1 |
| baseline_sn | symmetric_Tikh | -0.05379 | [-0.05966, -0.04793] | 3.4e-09 | 7.2e-08 |
| baseline_sn | symmetric_Tikh_SN | -0.0003129 | [-0.001565, 0.000939] | 0.61 | 1 |
| proposed_GP | proposed_GP_SN | -0.002715 | [-0.003601, -0.001828] | 8.3e-06 | 8.3e-05 |
| proposed_GP | proposed_Tikh | -0.05679 | [-0.06137, -0.05221] | 3.1e-10 | 8.7e-09 |
| proposed_GP | proposed_Tikh_SN | -0.004928 | [-0.005925, -0.003932] | 3.9e-08 | 5.1e-07 |
| proposed_GP | symmetric_Tikh | -0.05858 | [-0.06443, -0.05274] | 2.4e-09 | 5.6e-08 |
| proposed_GP | symmetric_Tikh_SN | -0.005104 | [-0.006125, -0.004083] | 4e-08 | 5.1e-07 |
| proposed_GP_SN | proposed_Tikh | -0.05408 | [-0.05867, -0.04948] | 3.1e-10 | 8.7e-09 |
| proposed_GP_SN | proposed_Tikh_SN | -0.002214 | [-0.003338, -0.001089] | 0.00064 | 0.0051 |
| proposed_GP_SN | symmetric_Tikh | -0.05587 | [-0.06172, -0.05001] | 2.9e-09 | 6.5e-08 |
| proposed_GP_SN | symmetric_Tikh_SN | -0.002389 | [-0.003534, -0.001244] | 0.00037 | 0.0034 |
| proposed_Tikh | proposed_Tikh_SN | +0.05186 | [0.04725, 0.05647] | 3.5e-10 | 8.7e-09 |
| proposed_Tikh | symmetric_Tikh | -0.001794 | [-0.008704, 0.005117] | 0.59 | 1 |
| proposed_Tikh | symmetric_Tikh_SN | +0.05169 | [0.04708, 0.0563] | 3.4e-10 | 8.7e-09 |
| proposed_Tikh_SN | symmetric_Tikh | -0.05366 | [-0.05952, -0.04779] | 3.7e-09 | 7.4e-08 |
| proposed_Tikh_SN | symmetric_Tikh_SN | -0.0001754 | [-0.001397, 0.001046] | 0.77 | 1 |
| symmetric_Tikh | symmetric_Tikh_SN | +0.05348 | [0.04762, 0.05934] | 3.7e-09 | 7.4e-08 |

## confirm_fresh_seeds / plantvillage_dcgan64

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp | 10 | 37.47 ± 2.9 | – | 0.02269 ± 0.0029 |
| zc_gp_sn | 10 | 40.03 ± 1.6 | – | 0.0246 ± 0.0015 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -2.562 | [-4.795, -0.3289] | 0.027 | 0.027 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -0.001912 | [-0.004159, 0.0003339] | 0.089 | 0.089 |

## gp_sweep_0.05 / cifar10_dcgan32

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp | 3 | 52.35 ± 3.9 | 5.591 ± 0.22 | 0.03693 ± 0.0036 |
| zc_gp_sn | 3 | 54.97 ± 0.51 | 5.57 ± 0.11 | 0.03898 ± 0.00061 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -2.62 | [-12.16, 6.921] | 0.37 | 0.37 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -0.002054 | [-0.01059, 0.006481] | 0.42 | 0.42 |

## gp_sweep_0.05 / plantvillage_dcgan64

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp | 3 | 40.57 ± 0.58 | – | 0.02658 ± 0.00054 |
| zc_gp_sn | 3 | 50.89 ± 5 | – | 0.03311 ± 0.0053 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -10.31 | [-22.56, 1.935] | 0.069 | 0.069 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -0.00653 | [-0.0194, 0.006336] | 0.16 | 0.16 |

## gp_sweep_0.2 / plantvillage_dcgan64

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp_sn | 3 | 47.47 ± 1.7 | – | 0.03023 ± 0.0021 |

## gp_sweep_0.5 / cifar10_dcgan32

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp | 3 | 51.2 ± 2.8 | 5.607 ± 0.19 | 0.03489 ± 0.003 |
| zc_gp_sn | 3 | 53.62 ± 1.1 | 5.639 ± 0.048 | 0.03694 ± 0.00093 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -2.422 | [-8.417, 3.573] | 0.27 | 0.27 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -0.002051 | [-0.008887, 0.004786] | 0.37 | 0.37 |

## gp_sweep_10.0 / cifar10_dcgan32

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp | 3 | 52.32 ± 0.67 | 5.525 ± 0.084 | 0.03689 ± 0.00076 |
| zc_gp_sn | 3 | 54.88 ± 1.1 | 5.55 ± 0.16 | 0.0387 ± 0.00085 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -2.557 | [-4.702, -0.4114] | 0.031 | 0.031 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -0.00181 | [-0.00364, 2.071e-05] | 0.052 | 0.052 |

## gp_sweep_10.0 / plantvillage_dcgan64

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp | 3 | 41.55 ± 2.9 | – | 0.02608 ± 0.0032 |
| zc_gp_sn | 3 | 44.04 ± 2.3 | – | 0.02787 ± 0.0016 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -2.483 | [-8.619, 3.653] | 0.32 | 0.32 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -0.001791 | [-0.008349, 0.004768] | 0.45 | 0.45 |

## gp_sweep_2.0 / cifar10_dcgan32

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp | 3 | 51.74 ± 1.1 | 5.653 ± 0.14 | 0.03571 ± 0.0012 |
| zc_gp_sn | 3 | 55.58 ± 1.7 | 5.491 ± 0.14 | 0.0388 ± 0.002 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -3.848 | [-7.288, -0.4086] | 0.037 | 0.037 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -0.00309 | [-0.007246, 0.001066] | 0.1 | 0.1 |

## gp_sweep_2.0 / plantvillage_dcgan64

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp | 3 | 37.32 ± 1.3 | – | 0.02243 ± 0.0014 |
| zc_gp_sn | 3 | 44.2 ± 2.6 | – | 0.02793 ± 0.0023 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -6.89 | [-12.37, -1.413] | 0.028 | 0.028 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| zc_gp | zc_gp_sn | -0.005504 | [-0.01028, -0.0007304] | 0.034 | 0.034 |

## gp_sweep_5.0 / plantvillage_dcgan64

| method | n | FID | IS | KID |
|---|---|---|---|---|
| zc_gp_sn | 3 | 40.4 ± 2.7 | – | 0.02447 ± 0.0026 |

## plantvillage_main / plantvillage_dcgan64

| method | n | FID | IS | KID |
|---|---|---|---|---|
| asym_tik | 10 | 77.69 ± 9.6 | – | 0.06351 ± 0.011 |
| asym_tik_sn | 10 | 52.62 ± 2.8 | – | 0.03462 ± 0.0027 |
| baseline_plain | 10 | 83.45 ± 7.9 | – | 0.07071 ± 0.0098 |
| baseline_sn | 10 | 53.08 ± 3.3 | – | 0.03499 ± 0.0031 |
| zc_gp | 10 | 36.39 ± 2.4 | – | 0.02219 ± 0.0025 |
| zc_gp_sn | 10 | 45.14 ± 1.8 | – | 0.02835 ± 0.0013 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| asym_tik | asym_tik_sn | +25.07 | [18.09, 32.05] | 9.3e-06 | 4.5e-05 |
| asym_tik | baseline_plain | -5.763 | [-14.02, 2.497] | 0.16 | 0.32 |
| asym_tik | baseline_sn | +24.61 | [17.57, 31.66] | 9e-06 | 4.5e-05 |
| asym_tik | zc_gp | +41.3 | [34.36, 48.25] | 1e-07 | 9.3e-07 |
| asym_tik | zc_gp_sn | +32.55 | [25.64, 39.45] | 1.3e-06 | 9.1e-06 |
| asym_tik_sn | baseline_plain | -30.83 | [-36.62, -25.03] | 1.3e-07 | 1e-06 |
| asym_tik_sn | baseline_sn | -0.4534 | [-3.317, 2.41] | 0.74 | 0.74 |
| asym_tik_sn | zc_gp | +16.24 | [13.8, 18.67] | 5.1e-11 | 7.6e-10 |
| asym_tik_sn | zc_gp_sn | +7.48 | [5.265, 9.695] | 2.6e-06 | 1.6e-05 |
| baseline_plain | baseline_sn | +30.38 | [24.5, 36.25] | 9.3e-08 | 9.3e-07 |
| baseline_plain | zc_gp | +47.07 | [41.31, 52.82] | 2.4e-09 | 3.1e-08 |
| baseline_plain | zc_gp_sn | +38.31 | [32.61, 44] | 3.7e-08 | 4.5e-07 |
| baseline_sn | zc_gp | +16.69 | [13.96, 19.42] | 4.9e-10 | 6.8e-09 |
| baseline_sn | zc_gp_sn | +7.933 | [5.38, 10.49] | 1.1e-05 | 4.5e-05 |
| zc_gp | zc_gp_sn | -8.757 | [-10.77, -6.746] | 6e-08 | 6.6e-07 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| asym_tik | asym_tik_sn | +0.02889 | [0.02105, 0.03673] | 8.6e-06 | 6.1e-05 |
| asym_tik | baseline_plain | -0.0072 | [-0.01691, 0.002514] | 0.14 | 0.27 |
| asym_tik | baseline_sn | +0.02852 | [0.02065, 0.03639] | 8.7e-06 | 6.1e-05 |
| asym_tik | zc_gp | +0.04132 | [0.03349, 0.04914] | 3.5e-07 | 3.9e-06 |
| asym_tik | zc_gp_sn | +0.03516 | [0.02741, 0.04291] | 2.4e-06 | 1.9e-05 |
| asym_tik_sn | baseline_plain | -0.03609 | [-0.04325, -0.02893] | 4e-07 | 3.9e-06 |
| asym_tik_sn | baseline_sn | -0.0003661 | [-0.003109, 0.002376] | 0.78 | 0.78 |
| asym_tik_sn | zc_gp | +0.01243 | [0.009942, 0.01492] | 4.5e-09 | 6.7e-08 |
| asym_tik_sn | zc_gp_sn | +0.006274 | [0.004202, 0.008345] | 2.1e-05 | 8.4e-05 |
| baseline_plain | baseline_sn | +0.03572 | [0.02853, 0.04291] | 3.6e-07 | 3.9e-06 |
| baseline_plain | zc_gp | +0.04852 | [0.04138, 0.05566] | 2.6e-08 | 3.4e-07 |
| baseline_plain | zc_gp_sn | +0.04236 | [0.0353, 0.04942] | 2e-07 | 2.4e-06 |
| baseline_sn | zc_gp | +0.0128 | [0.01014, 0.01545] | 1e-08 | 1.4e-07 |
| baseline_sn | zc_gp_sn | +0.00664 | [0.004349, 0.008931] | 3.9e-05 | 0.00012 |
| zc_gp | zc_gp_sn | -0.006157 | [-0.008092, -0.004221] | 1.1e-05 | 6.1e-05 |

## tik1e-3 / cifar10_dcgan32

| method | n | FID | IS | KID |
|---|---|---|---|---|
| asym_tik | 10 | 51.6 ± 2.8 | 5.674 ± 0.23 | 0.03746 ± 0.0021 |
| asym_tik_sn | 10 | 47.23 ± 1.7 | 5.656 ± 0.093 | 0.03508 ± 0.0015 |
| sym_tik | 10 | 52.23 ± 4.3 | 5.505 ± 0.34 | 0.03921 ± 0.0041 |
| sym_tik_sn | 10 | 50.58 ± 1.1 | 5.586 ± 0.16 | 0.03924 ± 0.002 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| asym_tik | asym_tik_sn | +4.372 | [2.151, 6.593] | 0.0008 | 0.004 |
| asym_tik | sym_tik | -0.631 | [-4.065, 2.803] | 0.7 | 0.79 |
| asym_tik | sym_tik_sn | +1.023 | [-1.08, 3.125] | 0.31 | 0.79 |
| asym_tik_sn | sym_tik | -5.003 | [-8.168, -1.838] | 0.0049 | 0.02 |
| asym_tik_sn | sym_tik_sn | -3.349 | [-4.714, -1.985] | 8.9e-05 | 0.00054 |
| sym_tik | sym_tik_sn | +1.654 | [-1.443, 4.75] | 0.26 | 0.79 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| asym_tik | asym_tik_sn | +0.00238 | [0.0006391, 0.004122] | 0.01 | 0.052 |
| asym_tik | sym_tik | -0.001748 | [-0.004873, 0.001377] | 0.25 | 0.5 |
| asym_tik | sym_tik_sn | -0.00178 | [-0.003724, 0.0001637] | 0.07 | 0.21 |
| asym_tik_sn | sym_tik | -0.004129 | [-0.007137, -0.001121] | 0.011 | 0.052 |
| asym_tik_sn | sym_tik_sn | -0.004161 | [-0.005837, -0.002484] | 7.1e-05 | 0.00042 |
| sym_tik | sym_tik_sn | -3.185e-05 | [-0.003132, 0.003068] | 0.98 | 0.98 |

## tik1e-3 / plantvillage_dcgan64

| method | n | FID | IS | KID |
|---|---|---|---|---|
| asym_tik | 10 | 61.38 ± 4.5 | – | 0.04744 ± 0.0049 |
| asym_tik_sn | 10 | 54.59 ± 2.4 | – | 0.03821 ± 0.0028 |

**Welch tests, FID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| asym_tik | asym_tik_sn | +6.787 | [3.335, 10.24] | 0.00089 | 0.00089 |

**Welch tests, KID**

| A | B | A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| asym_tik | asym_tik_sn | +0.009225 | [0.005388, 0.01306] | 0.00014 | 0.00014 |
