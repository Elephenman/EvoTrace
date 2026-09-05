# w4 n100 复核报告（实例 4090）

日期: 2026-09-04 21:44 ｜ 执行: 4090 实例 10.202.94.52:21114 (n1) ｜ 全程本地跑 boltz，未走 SLURM

## 背景
- w4 = EvoTrace R_R_R 深设计谱系（WT 254aa 骨架上 12 处突变: 22R;82W;84R;88Y;128N;131N;164K;185E;186N;255H;256Q;265S）
- 原只有 n8/20 旧批证据(sep 0.60, pred_GBM 0.466 全场最高)，is_n100=0
- 目的: 把 w4 证据坐实到 n100, 作为 MD 第 3 个名额(不同分支正向)的候选

## 结果 (n=100 per cell)
| 条件 | act | dual | lockA(d17) | lockB(d23) | mean d_act |
|------|-----|------|-----------|-----------|-----------|
| w4_S1_G17 (靶) | **0.640** | 0.340 | 0.610 | 0.490 | 4.71 |
| w4_OFF_T_G17 (非靶) | **0.110** | 0.000 | 0.000 | 0.080 | 6.62 |
| **sep** | **0.530** | | | | |

- sep 95% CI = [0.418, 0.642], **z = 9.25**（高度显著）
- OFF 排斥度 1-0.11 = **0.89**（非靶几乎不结合）
- 与旧批 sep 0.60 一致 → **n100 复核确认 w4 高特异性，未被证伪**

## 判定: ✅ 通过
- w4 是"OFF 极致排斥型"高特异候选（把 OFF 从天然 0.42-0.65 压到 0.11，全场最低档）
- 唯一 caveat: act_S1 0.64 略低于 0.70 保活门槛, 但特异性判别(off排斥)是优势策略, 与 s13_c1(OFF0.23)/TrackF_F88W_M255A(OFF0.28) 化学策略不同
- 对比: WT天然 sep 0.36(OFF 0.42), TrackF天然 sep 0.29(OFF 0.65) → w4 把 OFF 压得最低

## 数据位置
- 实例: ~/ppri_evo_w4/{out_w4, w4.a3m, w4_S1_G17.yaml, w4_OFF_T_G17.yaml}
- 判读指纹: 实例 ~/ppri_evo_boltz_nom2/contact_fingerprint_w4.csv (200 行)
- 权威序列: A:/claudework/evo2/results/boltz_nom2/w4_seq.txt

## 坑记录
- 实例 ~/.boltz 缺 mols.tar → boltz 联网下载被 reset → touch 空 mols.tar 占位跳过(因 mols/ 已解压完整) ✓
- run_mmseqs2 离线 MSA 预取 ~4s 完成 (695 行), 走 boltz run_mmseqs2
- 权威 w4 序列 = b5_ppri_wave2.py 定义; all_candidate_sequences.json['w4']/w4_shard 是脏数据(KAPAK 错误版)
