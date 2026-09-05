# DL v7 并行版 —— n=100 干净标签 + 变体专属表征

rows=2960, cells=74, systems=37, dev=cpu, epochs=24, threads=2

## V1 按系统 LOO（v4@n100 基准: act +0.739 / dual +0.747 / dcat —）

- y_act: **+0.770**
- y_dual: **+0.838**
- y_dcat: **+0.748**

## V1b sep（v4@n100 基准: rho +0.330 / 符号 0.92）

- 系统数=37, 全量 **-0.124**, 符号 **0.95**, top10 **-0.402**

## 对 v4@n100 基准的判决

- V1 act : v7 +0.770  vs 基准 +0.739  → ✓
- V1 dual: v7 +0.838  vs 基准 +0.747  → ✓
- V1 sep : v7 rho -0.124/符号 0.95  vs 基准 +0.330/0.92  → ✗
- **判决: REVIEW**（V1 act/dual 达基准且 sep 不弱于基准则为 PASS）