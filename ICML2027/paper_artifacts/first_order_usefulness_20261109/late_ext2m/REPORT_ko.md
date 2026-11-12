# 1차 유용성 측정 보고서: 후반부

설계와 판정 규칙: 이 폴더의 `PROTOCOL_late_ko.md`. 부호 규칙: 모든 gain은 `R(uniform 갱신 후) − R(arm 갱신 후)`이며 양수면 그 arm이 정책 조회 오차를 더 줄였다. 표의 gain은 R0 대비 백분율이다.

측정한 checkpoint: 30개 (학습 arm별: random 10, saber 10, td_topk 10). 실행 결과: success 20, failed 10, partial 0.

## 사전 고정 판정

단위는 실행(arm, seed) 하나다. 정확 단측 부호 검정 p ≤ 0.05이면 consistent. 수치적으로 null이거나 모든 배치에서 p = 0 또는 1이라 두 가중이 같은 실행은 tie로 세지 않는다.

| R | 층 | 검사 | 판정 | + | − | tie | p |
|---|---|---|---|---|---|---|---|
| R1 | success, all arms | H1 SABER > uniform (K=100) | mixed | 8 | 12 | 0 | 0.2517 |
| R1 | success, all arms | H1 SABER > uniform (K=10) | mixed | 10 | 10 | 0 | 0.5881 |
| R1 | success, all arms | H1 SABER > uniform (real 1-step) | mixed | 8 | 12 | 0 | 0.2517 |
| R1 | success, all arms | H1 SABER > uniform (G3 prediction) | mixed | 9 | 11 | 0 | 0.4119 |
| R1 | success, all arms | H2 SABER > Random (K=100) | mixed | 11 | 9 | 0 | 0.4119 |
| R1 | success, all arms | H2 SABER > TD top-k (K=100) | mixed | 9 | 11 | 0 | 0.4119 |
| R1 | success, all arms | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 10 | 10 | 0 | 0.5881 |
| R1 | success, all arms | H3 kappa(M) > kappa(E\M), dist-matched | consistent + | 20 | 0 | 0 | 0.0 |
| R1 | success, all arms | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 8 | 12 | 0 | 0.2517 |
| R1 | success, SABER-trained | H1 SABER > uniform (K=100) | mixed | 2 | 2 | 0 | 0.6875 |
| R1 | success, SABER-trained | H1 SABER > uniform (K=10) | mixed | 2 | 2 | 0 | 0.6875 |
| R1 | success, SABER-trained | H1 SABER > uniform (real 1-step) | mixed | 1 | 3 | 0 | 0.3125 |
| R1 | success, SABER-trained | H1 SABER > uniform (G3 prediction) | mixed | 2 | 2 | 0 | 0.6875 |
| R1 | success, SABER-trained | H2 SABER > Random (K=100) | mixed | 2 | 2 | 0 | 0.6875 |
| R1 | success, SABER-trained | H2 SABER > TD top-k (K=100) | mixed | 2 | 2 | 0 | 0.6875 |
| R1 | success, SABER-trained | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 2 | 2 | 0 | 0.6875 |
| R1 | success, SABER-trained | H3 kappa(M) > kappa(E\M), dist-matched | mixed | 4 | 0 | 0 | 0.0625 |
| R1 | success, SABER-trained | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 3 | 1 | 0 | 0.3125 |
| R1 | success, Random-trained | H1 SABER > uniform (K=100) | mixed | 3 | 5 | 0 | 0.3633 |
| R1 | success, Random-trained | H1 SABER > uniform (K=10) | mixed | 4 | 4 | 0 | 0.6367 |
| R1 | success, Random-trained | H1 SABER > uniform (real 1-step) | mixed | 4 | 4 | 0 | 0.6367 |
| R1 | success, Random-trained | H1 SABER > uniform (G3 prediction) | mixed | 4 | 4 | 0 | 0.6367 |
| R1 | success, Random-trained | H2 SABER > Random (K=100) | mixed | 4 | 4 | 0 | 0.6367 |
| R1 | success, Random-trained | H2 SABER > TD top-k (K=100) | mixed | 3 | 5 | 0 | 0.3633 |
| R1 | success, Random-trained | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 4 | 4 | 0 | 0.6367 |
| R1 | success, Random-trained | H3 kappa(M) > kappa(E\M), dist-matched | consistent + | 8 | 0 | 0 | 0.0039 |
| R1 | success, Random-trained | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 2 | 6 | 0 | 0.1445 |
| R1 | success, TD-trained | H1 SABER > uniform (K=100) | mixed | 3 | 5 | 0 | 0.3633 |
| R1 | success, TD-trained | H1 SABER > uniform (K=10) | mixed | 4 | 4 | 0 | 0.6367 |
| R1 | success, TD-trained | H1 SABER > uniform (real 1-step) | mixed | 3 | 5 | 0 | 0.3633 |
| R1 | success, TD-trained | H1 SABER > uniform (G3 prediction) | mixed | 3 | 5 | 0 | 0.3633 |
| R1 | success, TD-trained | H2 SABER > Random (K=100) | mixed | 5 | 3 | 0 | 0.3633 |
| R1 | success, TD-trained | H2 SABER > TD top-k (K=100) | mixed | 4 | 4 | 0 | 0.6367 |
| R1 | success, TD-trained | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 4 | 4 | 0 | 0.6367 |
| R1 | success, TD-trained | H3 kappa(M) > kappa(E\M), dist-matched | consistent + | 8 | 0 | 0 | 0.0039 |
| R1 | success, TD-trained | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 3 | 5 | 0 | 0.3633 |
| R1 | failed, all arms | H1 SABER > uniform (K=100) | mixed | 1 | 2 | 7 | 0.5 |
| R1 | failed, all arms | H1 SABER > uniform (K=10) | mixed | 1 | 2 | 7 | 0.5 |
| R1 | failed, all arms | H1 SABER > uniform (real 1-step) | mixed | 1 | 2 | 7 | 0.5 |
| R1 | failed, all arms | H1 SABER > uniform (G3 prediction) | mixed | 3 | 0 | 7 | 0.125 |
| R1 | failed, all arms | H2 SABER > Random (K=100) | mixed | 1 | 0 | 9 | 0.5 |
| R1 | failed, all arms | H2 SABER > TD top-k (K=100) | mixed | 0 | 1 | 9 | 0.5 |
| R1 | failed, all arms | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 1 | 2 | 7 | 0.5 |
| R1 | failed, all arms | H3 kappa(M) > kappa(E\M), dist-matched | mixed | 0 | 1 | 9 | 0.5 |
| R1 | failed, all arms | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 3 | 0 | 7 | 0.125 |
| R2 | success, all arms | H1 SABER > uniform (K=100) | mixed | 10 | 10 | 0 | 0.5881 |
| R2 | success, all arms | H1 SABER > uniform (K=10) | mixed | 12 | 8 | 0 | 0.2517 |
| R2 | success, all arms | H1 SABER > uniform (real 1-step) | mixed | 13 | 7 | 0 | 0.1316 |
| R2 | success, all arms | H1 SABER > uniform (G3 prediction) | mixed | 13 | 7 | 0 | 0.1316 |
| R2 | success, all arms | H2 SABER > Random (K=100) | mixed | 11 | 9 | 0 | 0.4119 |
| R2 | success, all arms | H2 SABER > TD top-k (K=100) | mixed | 13 | 7 | 0 | 0.1316 |
| R2 | success, all arms | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 11 | 9 | 0 | 0.4119 |
| R2 | success, all arms | H3 kappa(M) > kappa(E\M), dist-matched | consistent + | 20 | 0 | 0 | 0.0 |
| R2 | success, all arms | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 10 | 10 | 0 | 0.5881 |
| R2 | success, SABER-trained | H1 SABER > uniform (K=100) | mixed | 3 | 1 | 0 | 0.3125 |
| R2 | success, SABER-trained | H1 SABER > uniform (K=10) | mixed | 2 | 2 | 0 | 0.6875 |
| R2 | success, SABER-trained | H1 SABER > uniform (real 1-step) | mixed | 3 | 1 | 0 | 0.3125 |
| R2 | success, SABER-trained | H1 SABER > uniform (G3 prediction) | mixed | 4 | 0 | 0 | 0.0625 |
| R2 | success, SABER-trained | H2 SABER > Random (K=100) | mixed | 3 | 1 | 0 | 0.3125 |
| R2 | success, SABER-trained | H2 SABER > TD top-k (K=100) | mixed | 3 | 1 | 0 | 0.3125 |
| R2 | success, SABER-trained | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 4 | 0 | 0 | 0.0625 |
| R2 | success, SABER-trained | H3 kappa(M) > kappa(E\M), dist-matched | mixed | 4 | 0 | 0 | 0.0625 |
| R2 | success, SABER-trained | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 3 | 1 | 0 | 0.3125 |
| R2 | success, Random-trained | H1 SABER > uniform (K=100) | mixed | 2 | 6 | 0 | 0.1445 |
| R2 | success, Random-trained | H1 SABER > uniform (K=10) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, Random-trained | H1 SABER > uniform (real 1-step) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, Random-trained | H1 SABER > uniform (G3 prediction) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, Random-trained | H2 SABER > Random (K=100) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, Random-trained | H2 SABER > TD top-k (K=100) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, Random-trained | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 3 | 5 | 0 | 0.3633 |
| R2 | success, Random-trained | H3 kappa(M) > kappa(E\M), dist-matched | consistent + | 8 | 0 | 0 | 0.0039 |
| R2 | success, Random-trained | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 2 | 6 | 0 | 0.1445 |
| R2 | success, TD-trained | H1 SABER > uniform (K=100) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, TD-trained | H1 SABER > uniform (K=10) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, TD-trained | H1 SABER > uniform (real 1-step) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, TD-trained | H1 SABER > uniform (G3 prediction) | mixed | 4 | 4 | 0 | 0.6367 |
| R2 | success, TD-trained | H2 SABER > Random (K=100) | mixed | 3 | 5 | 0 | 0.3633 |
| R2 | success, TD-trained | H2 SABER > TD top-k (K=100) | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | success, TD-trained | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 4 | 4 | 0 | 0.6367 |
| R2 | success, TD-trained | H3 kappa(M) > kappa(E\M), dist-matched | consistent + | 8 | 0 | 0 | 0.0039 |
| R2 | success, TD-trained | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 5 | 3 | 0 | 0.3633 |
| R2 | failed, all arms | H1 SABER > uniform (K=100) | mixed | 2 | 1 | 7 | 0.5 |
| R2 | failed, all arms | H1 SABER > uniform (K=10) | mixed | 1 | 2 | 7 | 0.5 |
| R2 | failed, all arms | H1 SABER > uniform (real 1-step) | mixed | 1 | 2 | 7 | 0.5 |
| R2 | failed, all arms | H1 SABER > uniform (G3 prediction) | mixed | 3 | 0 | 7 | 0.125 |
| R2 | failed, all arms | H2 SABER > Random (K=100) | mixed | 1 | 0 | 9 | 0.5 |
| R2 | failed, all arms | H2 SABER > TD top-k (K=100) | mixed | 1 | 0 | 9 | 0.5 |
| R2 | failed, all arms | inverse (psi=0 upweighted) > uniform (K=100) | mixed | 1 | 2 | 7 | 0.5 |
| R2 | failed, all arms | H3 kappa(M) > kappa(E\M), dist-matched | mixed | 0 | 1 | 9 | 0.5 |
| R2 | failed, all arms | dir: cos(M) > cos(not M), abs(delta)-matched | mixed | 3 | 0 | 7 | 0.125 |

## Checkpoint별 값 (R1, gain / R0)

| arm | seed | step | 마지막 평가 | p | live | clip | √R0 | 1-step SABER | K=10 SABER | K=100 SABER | K=100 Random | K=100 TD | K=100 역방향 | κ gap | G3 부호 일치 | query phase |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 11 | 2,000,000 | 1.00 | 0.656 | 64/64 | 1.00 | 1.46e+00 | -0.18% | -1.96% | +3.11% | +0.24% | -1.61% | -8.89% | +0.010 | 0.83 | 1091/545/263/2101 |
| random | 12 | 2,000,000 | 1.00 | 0.482 | 64/64 | 1.00 | 1.21e+00 | +0.19% | -3.66% | -0.31% | -0.98% | +0.28% | +2.96% | +0.018 | 0.97 | 748/233/235/2784 |
| random | 13 | 2,000,000 | 1.00 | 0.520 | 64/64 | 1.00 | 4.05e-01 | +0.55% | -1.08% | +0.07% | +3.99% | -98.33% | -13.63% | +0.005 | 1.00 | 456/165/168/3211 |
| random | 14 | 2,000,000 | 0.00 | 0.991 | 55/64 | 0.00 | 1.96e-02 | -1.52% | -281.53% | -13.93% | -13.93% | -13.93% | -40.76% | – | 0.09 | 4000/0/0/0 |
| random | 15 | 2,000,000 | 1.00 | 0.549 | 64/64 | 1.00 | 1.16e+00 | -0.07% | +1.62% | -2.78% | +6.12% | +1.17% | -23.67% | +0.010 | 0.95 | 534/299/839/2328 |
| random | 16 | 2,000,000 | 1.00 | 0.514 | 64/64 | 1.00 | 4.82e-01 | -0.32% | -10.74% | -328.05% | +126.04% | +19.77% | +103.92% | +0.007 | 0.88 | 569/221/194/3016 |
| random | 17 | 2,000,000 | 1.00 | 0.554 | 64/64 | 1.00 | 5.77e-01 | +0.25% | +7.80% | -17.34% | -19.27% | -7.55% | -52.25% | +0.010 | 0.94 | 651/197/180/2972 |
| random | 18 | 2,000,000 | 1.00 | 0.444 | 64/64 | 1.00 | 1.16e+00 | +0.28% | +30.08% | -1.65% | -2.29% | -2.07% | +0.64% | +0.022 | 0.98 | 852/337/300/2511 |
| random | 19 | 2,000,000 | 0.00 | 0.595 | 64/64 | 0.00 | 2.06e-02 | +87.46% | -24.50% | +164.13% | +138.05% | +172.43% | -126.16% | -0.013 | 1.00 | 4000/0/0/0 |
| random | 20 | 2,000,000 | 1.00 | 0.566 | 64/64 | 1.00 | 7.19e-01 | -1.11% | +8.63% | +4.89% | +29.17% | +21.14% | +30.52% | +0.017 | 0.98 | 410/349/142/3099 |
| saber | 11 | 2,000,000 | 1.00 | 0.614 | 64/64 | 1.00 | 5.18e-01 | +1.09% | -7.65% | +63.89% | -9.06% | -23.78% | +81.93% | +0.002 | 0.95 | 479/177/172/3172 |
| saber | 12 | 2,000,000 | 1.00 | 0.504 | 64/64 | 1.00 | 1.64e+00 | -0.05% | -1.64% | +16.32% | +16.13% | +9.82% | +21.63% | +0.030 | 0.98 | 649/202/210/2939 |
| saber | 13 | 2,000,000 | 0.00 | 1.000 | 0/64 | 0.00 | 1.76e-03 | – | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | – | – | 4000/0/0/0 |
| saber | 14 | 2,000,000 | 0.00 | 1.000 | 0/64 | 0.00 | 9.24e-03 | – | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | – | – | 4000/0/0/0 |
| saber | 15 | 2,000,000 | 0.00 | 0.965 | 64/64 | 0.00 | 3.57e-03 | -10.44% | +25.71% | -18.86% | -18.86% | -18.86% | +61.01% | – | 0.00 | 4000/0/0/0 |
| saber | 16 | 2,000,000 | 0.00 | 1.000 | 0/64 | 0.00 | 6.53e-04 (null) | – | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | – | – | 4000/0/0/0 |
| saber | 17 | 2,000,000 | 1.00 | 0.576 | 64/64 | 1.00 | 4.37e-01 | -0.15% | +0.61% | -3.77% | -1.27% | -0.56% | -7.65% | +0.022 | 0.64 | 366/244/190/3200 |
| saber | 18 | 2,000,000 | 0.00 | 0.986 | 59/64 | 0.00 | 5.02e-04 (null) | +0.03% | -15.48% | -124.63% | -124.63% | -124.63% | +408.58% | – | 0.68 | 4000/0/0/0 |
| saber | 19 | 2,000,000 | 1.00 | 0.483 | 64/64 | 1.00 | 9.41e-01 | -0.42% | +18.85% | -22.71% | -14.90% | -7.55% | -23.37% | +0.030 | 0.92 | 828/172/177/2823 |
| saber | 20 | 2,000,000 | 0.00 | 0.986 | 62/64 | 0.00 | 5.74e-04 (null) | +2.89% | -4083.31% | -8074.94% | -8074.94% | -8074.94% | +2517.20% | – | 0.97 | 4000/0/0/0 |
| td_topk | 11 | 2,000,000 | 1.00 | 0.598 | 64/64 | 1.00 | 1.09e+00 | +0.47% | +25.69% | -22.53% | -0.62% | +9.75% | +4.82% | +0.027 | 0.91 | 907/200/195/2698 |
| td_topk | 12 | 2,000,000 | 1.00 | 0.473 | 64/64 | 1.00 | 8.51e-01 | -0.19% | -3.71% | -1.06% | -3.22% | -5.45% | -5.66% | +0.008 | 0.91 | 484/290/210/3016 |
| td_topk | 13 | 2,000,000 | 0.00 | 1.000 | 0/64 | 0.00 | 5.64e-04 (null) | – | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | – | – | 4000/0/0/0 |
| td_topk | 14 | 2,000,000 | 1.00 | 0.461 | 64/64 | 1.00 | 3.25e-01 | +1.36% | +10.69% | -41.99% | +8.45% | +8.67% | -0.81% | +0.012 | 0.94 | 345/274/147/3234 |
| td_topk | 15 | 2,000,000 | 1.00 | 0.536 | 64/64 | 1.00 | 2.41e-01 | -2.32% | +21.96% | +88.71% | +105.59% | +31.68% | +89.57% | +0.018 | 0.80 | 264/272/147/3317 |
| td_topk | 16 | 2,000,000 | 0.00 | 0.992 | 55/64 | 0.00 | 5.55e-04 (null) | +1.53% | +168.22% | +2328.87% | +2328.87% | +2328.87% | -13247.07% | – | 0.96 | 4000/0/0/0 |
| td_topk | 17 | 2,000,000 | 1.00 | 0.604 | 64/64 | 1.00 | 5.87e-01 | +0.21% | +4.40% | +22.36% | -2.94% | -5.46% | +2.48% | +0.012 | 0.95 | 449/181/197/3173 |
| td_topk | 18 | 2,000,000 | 1.00 | 0.417 | 64/64 | 1.00 | 4.97e-01 | -1.36% | -11.39% | +31.51% | -4.47% | +15.42% | +18.45% | +0.020 | 0.98 | 347/215/207/3231 |
| td_topk | 19 | 2,000,000 | 1.00 | 0.543 | 64/64 | 1.00 | 1.12e+00 | -0.00% | -1.03% | -6.24% | -8.18% | -1.44% | -8.40% | +0.016 | 0.45 | 749/521/132/2598 |
| td_topk | 20 | 2,000,000 | 1.00 | 0.494 | 64/64 | 1.00 | 2.92e-01 | -0.28% | -12.77% | -13.94% | -24.05% | -10.20% | -22.29% | +0.020 | 0.67 | 242/252/177/3329 |

## Checkpoint별 값 (R2, gain / R0)

| arm | seed | step | 마지막 평가 | p | live | clip | √R0 | 1-step SABER | K=10 SABER | K=100 SABER | K=100 Random | K=100 TD | K=100 역방향 | κ gap | G3 부호 일치 | query phase |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 11 | 2,000,000 | 1.00 | 0.656 | 64/64 | 1.00 | 1.01e+00 | -0.25% | -1.85% | -0.06% | -0.57% | -1.94% | -5.51% | +0.010 | 0.86 | 181/93/105/645 |
| random | 12 | 2,000,000 | 1.00 | 0.482 | 64/64 | 1.00 | 1.58e+00 | +0.08% | -0.71% | +6.88% | -2.06% | -1.06% | -0.51% | +0.018 | 0.98 | 150/59/89/726 |
| random | 13 | 2,000,000 | 1.00 | 0.520 | 64/64 | 1.00 | 9.25e-01 | +0.02% | +0.38% | -0.53% | +9.02% | -10.91% | -22.81% | +0.005 | 0.84 | 181/48/58/737 |
| random | 14 | 2,000,000 | 0.00 | 0.991 | 55/64 | 0.00 | 8.29e-03 | -1.87% | -327.79% | -16.37% | -16.37% | -16.37% | -50.61% | – | 0.07 | 203/0/0/821 |
| random | 15 | 2,000,000 | 1.00 | 0.549 | 64/64 | 1.00 | 1.14e+00 | +0.29% | +1.72% | +4.52% | +3.66% | +5.81% | -30.97% | +0.010 | 0.92 | 176/65/77/706 |
| random | 16 | 2,000,000 | 1.00 | 0.514 | 64/64 | 1.00 | 9.29e-01 | -0.46% | -9.83% | -94.23% | +57.75% | +18.62% | +54.64% | +0.007 | 0.89 | 194/94/88/648 |
| random | 17 | 2,000,000 | 1.00 | 0.554 | 64/64 | 1.00 | 4.52e-01 | +0.61% | +9.40% | -4.23% | -7.43% | -12.14% | -53.99% | +0.010 | 0.97 | 160/29/57/778 |
| random | 18 | 2,000,000 | 1.00 | 0.444 | 64/64 | 1.00 | 1.23e+00 | +0.37% | +11.68% | -5.16% | -6.48% | -11.85% | +9.09% | +0.022 | 0.94 | 197/70/102/655 |
| random | 19 | 2,000,000 | 0.00 | 0.595 | 64/64 | 0.00 | 8.92e-03 | +96.91% | -31.21% | +145.69% | +119.74% | +140.31% | -103.13% | -0.013 | 1.00 | 203/0/0/821 |
| random | 20 | 2,000,000 | 1.00 | 0.566 | 64/64 | 1.00 | 7.20e-01 | -1.24% | +8.70% | -1.26% | +13.81% | +14.28% | +17.12% | +0.017 | 0.91 | 90/97/72/765 |
| saber | 11 | 2,000,000 | 1.00 | 0.614 | 64/64 | 1.00 | 4.41e-01 | +2.98% | -47.91% | +36.55% | +30.46% | -29.43% | +109.25% | +0.002 | 0.98 | 131/36/58/799 |
| saber | 12 | 2,000,000 | 1.00 | 0.504 | 64/64 | 1.00 | 1.34e+00 | +0.00% | -0.72% | +17.35% | +10.34% | +12.79% | +11.04% | +0.030 | 0.91 | 186/69/65/704 |
| saber | 13 | 2,000,000 | 0.00 | 1.000 | 0/64 | 0.00 | 3.34e-03 | – | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | – | – | 203/0/0/821 |
| saber | 14 | 2,000,000 | 0.00 | 1.000 | 0/64 | 0.00 | 3.57e-03 | – | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | – | – | 203/0/0/821 |
| saber | 15 | 2,000,000 | 0.00 | 0.965 | 64/64 | 0.00 | 1.89e-03 | -5.85% | +12.26% | +3.95% | +3.95% | +3.95% | +34.80% | – | 0.02 | 203/0/0/821 |
| saber | 16 | 2,000,000 | 0.00 | 1.000 | 0/64 | 0.00 | 2.59e-04 (null) | – | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | – | – | 203/0/0/821 |
| saber | 17 | 2,000,000 | 1.00 | 0.576 | 64/64 | 1.00 | 3.90e-01 | +0.07% | +2.92% | -24.63% | +6.83% | +4.04% | +7.73% | +0.022 | 0.72 | 89/69/58/808 |
| saber | 18 | 2,000,000 | 0.00 | 0.986 | 59/64 | 0.00 | 2.08e-04 (null) | +0.20% | -17.67% | -119.02% | -119.02% | -119.02% | +372.32% | – | 0.78 | 203/0/0/821 |
| saber | 19 | 2,000,000 | 1.00 | 0.483 | 64/64 | 1.00 | 8.38e-01 | -0.23% | +12.34% | +1.70% | -21.59% | -70.45% | +27.78% | +0.030 | 0.94 | 158/72/74/720 |
| saber | 20 | 2,000,000 | 0.00 | 0.986 | 62/64 | 0.00 | 2.79e-04 (null) | +7.66% | -3074.45% | -9562.68% | -9562.68% | -9562.68% | +7005.69% | – | 0.97 | 203/0/0/821 |
| td_topk | 11 | 2,000,000 | 1.00 | 0.598 | 64/64 | 1.00 | 8.51e-01 | +0.27% | +21.13% | +6.93% | +4.30% | +3.28% | +30.46% | +0.027 | 0.88 | 166/72/64/722 |
| td_topk | 12 | 2,000,000 | 1.00 | 0.473 | 64/64 | 1.00 | 7.70e-01 | +0.11% | -25.33% | -59.62% | -48.42% | -96.46% | -8.70% | +0.008 | 0.81 | 127/74/80/743 |
| td_topk | 13 | 2,000,000 | 0.00 | 1.000 | 0/64 | 0.00 | 2.77e-04 (null) | – | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | – | – | 203/0/0/821 |
| td_topk | 14 | 2,000,000 | 1.00 | 0.461 | 64/64 | 1.00 | 3.25e-01 | +1.55% | +12.60% | -63.70% | -3.05% | -4.64% | -6.56% | +0.012 | 0.98 | 92/55/52/825 |
| td_topk | 15 | 2,000,000 | 1.00 | 0.536 | 64/64 | 1.00 | 5.22e-01 | -0.61% | +5.02% | +2.64% | +21.13% | +10.06% | +19.36% | +0.018 | 0.70 | 88/65/55/816 |
| td_topk | 16 | 2,000,000 | 0.00 | 0.992 | 55/64 | 0.00 | 2.50e-04 (null) | +1.42% | +173.37% | +2831.69% | +2831.69% | +2831.69% | -15273.24% | – | 0.95 | 203/0/0/821 |
| td_topk | 17 | 2,000,000 | 1.00 | 0.604 | 64/64 | 1.00 | 4.60e-01 | +0.70% | +2.06% | -15.59% | -8.25% | -4.12% | -16.69% | +0.012 | 1.00 | 88/39/53/844 |
| td_topk | 18 | 2,000,000 | 1.00 | 0.417 | 64/64 | 1.00 | 8.02e-01 | -0.53% | -2.76% | +16.60% | +6.24% | +15.33% | +19.01% | +0.020 | 0.94 | 147/88/69/720 |
| td_topk | 19 | 2,000,000 | 1.00 | 0.543 | 64/64 | 1.00 | 6.42e-01 | +0.20% | -1.18% | +5.00% | +10.09% | -1.61% | +2.23% | +0.016 | 0.84 | 130/98/66/730 |
| td_topk | 20 | 2,000,000 | 1.00 | 0.494 | 64/64 | 1.00 | 9.98e-01 | -0.06% | +1.99% | +7.32% | +3.39% | +4.91% | -3.97% | +0.020 | 0.69 | 124/56/66/778 |

## 읽을 때 주의

- 독립 단위는 학습 실행이다. 같은 seed의 세 arm은 초기화와 첫 10k 무작위 행동을 공유한다.
- Random·TD로 학습한 checkpoint의 ψ는 'SABER였다면 무엇을 키웠을까'라는 반사실 값이다.
- live는 0 < p < 1인 배치 수다. 0이면 SABER 가중이 uniform과 같아 모든 gain이 0이다.
- (null)은 정책 조회 RMS residual이 float32 한계 아래라 부호를 세지 않은 checkpoint다.
- 고정 target 기준이고 Q^π 정확도가 아니다. 값 오차 개선은 ∇_a Q 개선을 뜻하지 않는다.
- actor와 target을 고정하고 critic만 갱신한 결과다.
