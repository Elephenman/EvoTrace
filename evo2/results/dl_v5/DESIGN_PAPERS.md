# 三篇文献方法学 → PprI 深度学习方案映射

日期: 2026-09-03 ｜ 脚本: `evo2/benchmarks/train_dl_v5.py`（v5）+ `train_sep_model_v4.py`（v4 基线）

## 1. 三篇论文的配方拆解

### p1 — EvoMax（Nat Biotech 2026, Wan/Gold/…/Gao）
"Adaptive model-guided protein evolution with sparse data optimizes compact eukaryotic genome editors"
- **场景**：Fanzor2 紧凑编辑器，稀疏标签（**209 个单点突变**的实测活性）
- **模型**：① 转移学习 GPR（自定义 BLOSUM62 相似度核，同位置 δ 门控）学"蛋白专属活性景观"；
  ② ESM-2 650M masked-mutant 概率作进化先验；③ ESM-IF（142M）+ AF3 结构做结构条件过滤
- **融合**：median-IQR 归一 → 加权几何组合（初轮 GPR 10% / ESM-IF 90%；
  随轮次背景优化后权重移向 ESM-2 促进探索）→ 迭代轮（提名→实测→回流）
- **关键承认**：单点层面运行，**不显式建模上位性**（limitation 原文）
- **结果**：FanzMAX v3-hLa 最高位点 97%，19 位点均值 ~33%，超 enNlovFz2/enCnCas12f1 2.6×

### p2 — ProMEP Cas9（Mol Syst Biol 2025, Wei/Cheng/…/Wei）
"AI-guided Cas9 engineering provides an effective strategy to enhance base editing"
- **场景**：Cas9 碱基编辑效率，**零样本打分起步**
- **模型**：ProMEP（~1.6 亿蛋白预训练的序列-结构上下文模型）全基因组单点饱和打分
- **流程**：top-18 单点实测（8 个正向）→ 优背景下对 **2.8×10¹² 组合库**全打分 →
  top-10 八点组合（AI-8.1~8.10）→ AI-8.3 平均 2-3×
- **关键教训**：PLM 偏置——天然 Cas9 同源序列 K 富集 → ProMEP 系统性偏好 X→K；
  需用同源序列统计显式校正/警示

### p3 — AI-redesigned starting points（Nature 2026, Krasnow/…/Liu）
"AI-redesigned starting points and outcomes enhance protein evolution"
- **场景**：BoNT 蛋白酶 PACE 进化
- **模型**：ProteinMPNN 全序列重设计作为**进化起点**（不训任务模型）
- **核心发现**：重设计起点提升突变鲁棒性 → 同选择压下进化出更高活性；
  重设计解锁 WT 背景不可达的序列（重设计进化出的突变在 WT 上失效——上位性实证）；
  **稳定性余量决定进化可达性**；特异性进化 79×（ataxin-2）

## 2. 三篇的共同结构 vs 我们已有的

| 要素 | p1 | p2 | p3 | 我们已有 |
|---|---|---|---|---|
| 大预训练模型做表征/先验 | ESM-2 + ESM-IF | ProMEP(160M 蛋白) | ProteinMPNN | ESM3-open 1.4B 嵌入（本地 218 个 npz）+ v3 DeepSet（ProteinGym 217 集训练） |
| 蛋白专属小模型在稀疏标签上训练 | GPR@209 标签 | —（纯零样本） | — | **v4 GBM/Ridge @151 cells×63 系统（ρ 0.64-0.72）** |
| 迭代轮 + 标签回流 | EvoMax 轮次 | 单轮组合提名 | PACE 连续进化 | Boltz 三轮（六候选/去混杂/V2-A）675 逐模型行 |
| 条件维度（核酸） | ωRNA 脚手架工程 | gRNA | ωRNA | **DNA 靶标条件（S1/OFF/GCA）——v4 消融证明是全部增益来源** |
| 上位性 | 明示不建模 | 组合打分（零样本） | 实证存在（重设计突变 WT 上失效） | **v4 GBM 交互项命中 V2-A 翻转方向** |
| 稳定性 | — | — | 进化可达性决定因素 | wave-2 M1 机制（ΔΔG 红旗归零） |

## 3. PprI DL v5 落地（本轮实现）

**架构**（`train_dl_v5.py`）:
- 双通路：位点 token transformer（53 token × [wt|mut|位置|is_mut|簇|ESM3 上下文×0.25|z]，
  d=32×2 层，DNA 条件 query 注意力池化）**+ 全局特征直连通路**（v4 的 160 维 GBM 特征，
  含热点 AA one-hot/电荷计数/nt10,17,23——绕过注意力稀释，v1 教训）
- 多任务头：act/dual（BCE）+ dcat/d17/d23（训练折标准化 MSE），逐模型行=标签噪声增广
- act 头偏置初始化为基率 logit（防常数捷径）；梯度裁剪 1.0

**训练数据**（零新算力，全部既有标注）:
- 旧战役 per_model_metrics 3240 行（每 cell 降采样 3 模型）+ evo2 三轮指纹 675 行 + v3cand 16 行
  = 913 行 / 63 系统 / 3 条件

**验证口径**（与 v4 严格同口径）:
- V1 按系统 LOO（cell 级 act/dual/dcat）
- V1b sep = act(S1)−act(OFF)（工具本职）+ top-|sep| 分层
- V3 V2-A 留出上位性（TrackF+M255I vs HQL2+M255I 翻转方向，8 seeds 集成）

## 4. 三篇带来的下一步路线（按性价比）

1. **组合空间全打分**（p2 核心）：v5 模型对 53 位点 × 20 AA × s13_c1/TrackF 背景
   的组合空间打分 → 提名 top-k 给 Boltz 复核——这是"预测对 PprI 有帮助"的直接形态
2. **集成方差获取**（p1 核心）：8 seeds σ → UCB = μ + κσ 提名下一轮 Boltz 批次
   （探索-开发权重随轮次移动）
3. **PLM 偏置审计**（p2 教训）：检查 v5/v4 对 X→K/R 的系统性偏好，
   用 PprI 同源序列（Deinococcus/Thermus）统计校正
4. **起点鲁棒性**（p3 核心）：ProteinMPNN/ESM-IF 对候选做结构条件过滤，
   稳定性余量作为特异性突变的"预算"（与 wave-2 M1 机制同构）
5. **逐变体 ESM3 嵌入**（升级表征）：对 63 系统 + 新候选跑集群 ESM3（esm3_embed_cluster.py
   现成），替换 WT 上下文 → 变体专属表征（p1 的 ESM-2 用法）
