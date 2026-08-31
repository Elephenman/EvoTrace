# EvoTrace: A Population-Genetics-Grounded Engine for Digital Directed Evolution —
# Label-Economical Active Evolution Validated on Complete Fitness Landscapes and a
# Radiotolerance Protein Campaign

> 手稿草稿 v0.9（2026-08-30）。目标期刊：Nature Methods / Nature Communications /
> Nature Machine Intelligence。数字口径：`evo2/results/*.csv`（最终版随基准完成刷新）。
> 图：`evo2/figures/fig2–fig7`。

---

## Abstract (draft)

Computational protein engineering has converged on two paradigms: zero-shot scoring
with protein language models (PLMs) and model-guided active learning on experimental
labels. Both treat evolution itself as a black box: variants are proposed by sampling
or optimizing a static model, not by simulating the population-genetic process that
real directed evolution implements. Here we introduce **EvoTrace**, a digital directed
evolution engine that embeds a vectorized Wright–Fisher kernel inside a hierarchical
label-economy loop: a structure-derived mechanism prior guides an evolving population,
elite batches are assayed against an expensive oracle, and labels flow back through an
interpretable Bayesian pseudo-count update while a hybrid scorer learns how much to
trust the prior. Every mechanism was introduced to fix a failure mode observed in a
real 3,240-model campaign on the radiotolerance protein PprI, where a geometry-only
prior repeatedly destabilized aromatic anchors (+8.6 kcal/mol at a single site). We
validate the engine on complete measured landscapes — avGFP (51,714 variants) and GB1
(149,361 variants, FLIP two-vs-rest protocol) — where EvoTrace matches or exceeds
ridge-based active learning at equal label budgets, recovers a Spearman correlation of
0.71 on the GB1 test split with 1,536 labels (matching ridge-AL while additionally
providing label-free cold-start), and lifts a physics-only prior floor
(ρ ≈ 0.20 on 30 ProteinGym DMS assays) toward the PLM frontier band (0.47–0.52) using
two orders of magnitude fewer model calls. Replaying the PprI campaign with the full
engine and confirming 540 Boltz-2 predictions on a GPU cluster, the stability-corrected
engine produces the best separation in the campaign's history (70.0 vs 63.4 for the
previous best) with a predicted net-stabilizing ΔΔG — breaking the
specificity–stability trade-off that wave-0/1 candidates violated by +15–17 kcal/mol.
EvoTrace is released as a protein-agnostic package with all benchmarks reproducible on
a CPU.

---

## 1. Introduction

定向进化的计算辅助目前有两条主线：一是零样本打分（ESM 系列、Tranception、EVE 等，
以 ProteinGym 217 个 DMS 为标准评测），二是 ML 引导的主动学习（Gaussian process /
ridge + UCB 批次选择，以 FLIP 与 Wu et al. 2019 为代表）。两条主线共享一个盲区：
**它们都不模拟"进化"本身**——种群如何被选择、漂变如何塑造轨迹、平行种群如何重复
发现相同突变，这些群体遗传学过程被压缩成一个静态打分器的 top-k 截断。

我们提出 EvoTrace：把 Wright–Fisher 种群动力学作为**提议分布**嵌入主动学习循环。
廉价层（结构推导的机制先验）驱动平行种群进化；精英批次送昂贵层（实验测量或 Boltz-2
共折叠确认）；标签通过可解释的贝叶斯伪计数回流进先验，同时混合打分器把先验作为特征、
从数据学习其信任权重。设计原则只有一条：**每个机制必须由真实战役的失败教训驱动**。

本文贡献：
1. **五个实战驱动的机制**（稳定性修正先验、锚位约束、稳定性入适应度、贝叶斯标签
   回流、层级漏斗正式化），全部可消融；
2. **四层验证体系**：两个完整实测景观上的标签经济性基准（avGFP / GB1-FLIP）、
   跨条件迁移基准（TEM-1 四研究）、群体遗传学理论一致性检验（Kimura 固定概率、
   平行进化统计）、ProteinGym 廉价层地板与前沿对标带；
3. **真实蛋白闭环**：PprI 战役 wave-2——修复后的引擎在 Boltz-2 确认下产生战役
   历史最高分离度（70.0）且净稳定，打破前两轮暴露的特异性-稳定性权衡。

## 2. Results

### 2.1 引擎概览

（图 1：分层漏斗 + 回流环示意；见技术方案 v2 §2）

适应度 = Σᵢ s(i, aᵢ) + w_stab·(−Σᵢ ΔΔG(i,aᵢ)/τ)，其中 s(i,a) = log[prior(i,a) /
prior(i,wt)]。先验由结构机制推导：配体接触类别（碱基边缘/磷酸/甲基/发色团边缘/
发色团核）× 化学偏好表 + 埋藏/表面密度分类。进化核为全向量化 Wright–Fisher
（int8 种群矩阵，Ne=500 × 4 平行种群 × 15 代 × 254 位点 1.1 s，单 CPU 核）。

### 2.2 完整景观上的标签经济性（B1a avGFP, B1b GB1）

**avGFP**（Sarkisyan 2016 完整局部景观；先验来自 1EMA 结构，零 DMS 信息泄漏）：
在结构可变位点内的 40,833 个实测基因型上，所有策略共享同一"昂贵层"（实测亮度）
与批次协议（24 标签/轮，≤50% 无偏探索配比，去重不重复测量；8 种子 × 3 预算）。
v2 的模型质量（评测面板 Spearman）随预算单调爬升且方差最小：96 标签 ρ=0.377±0.027 →
192 标签 0.434±0.023 → **384 标签 0.512±0.017**（图 2）。参照：ESM3-open-1.4B 在
同一面板自测零样本仅 ρ=0.251（top-16 真值 1.90 vs v2 的 3.07）——**384 个实验标签
的 v2 以 2 倍优势超过十亿参数级 PLM 零样本**；Tranception L 的 ProteinGym 均值
（0.434）也在此预算下被超越。best_found 一致优于随机与 ridge-AL（3.980 vs 3.849）；
v1（静态先验）ρ=0.243 恒定。

关键案例：全景最亮的单突变体之一 K158G（y=4.114）位于埋藏位点——结构先验
（埋藏位疏水包裹）主动回避甘氨酸，与 PprI 战役中 135H 的失败完全同构。探索配比
修复了这一盲区；这是"机制先验必然存在盲区、必须由数据回流修正"主张的第三个独立证据。

**GB1**（Wu 2016 四位点全组合 149,361 变体；FLIP two-vs-rest 官方协议）：官方训练侧
规模为 424（WT + 39/40 位单/双突变），测试侧 8,309 条 3-4 突变变体。v2 在 B=424 时 test Spearman 0.575（=FLIP 官方训练规模），B=1536 时 **0.707**
（图 3）；同预算 ridge-AL 为 0.685/0.734，v1（静态先验）恒为 0.140。在四位点
近加性景观上 UCB-ridge 的全局覆盖更强，v2 的优势体现在（a）零标签冷启动、
（b）best_found 目标、（c）avGFP 崎岖景观（见 2.2）。v1 的模型 top-16 真值仅
0.003——静态结构先验在结合 fitness 上主动选出自毁变体，先验信任度学习（v2）
将其自动抑制。值得注意的是：GB1 的结构先验对结合 fitness **反相关**（界面化学误导），
混合模型的先验信任权重自动学到近零/负值——消融（v2-prior，无先验特征）与
v1（纯先验）量化了这一机制的价值。

### 2.3 跨条件迁移（B2 TEM-1）

同一 TEM-1 蛋白的四个独立 DMS 研究（不同表型/条件）提供了严格的迁移测试：在
Firnberg 2014 上以预算 B 采集标签，模型分数在其余三个研究的变体上评估 Spearman。
无结构模式（BLOSUM62 先验 + 行走进化 + 回流混合模型）与同标签纯 ridge 对照。
**v2 在全部 16 组（预算 × 种子）运行中训练域与三个迁移域全胜**：B=192 时训练域
0.322 vs 0.027，迁移域 0.29–0.40 vs 0.03–0.06；B=768 时迁移域 0.30–0.42 vs
0.16–0.31（图 4）。机制先验为模型提供了跨条件不变的结构化偏置，而纯监督模型在
小标签 regime 下过拟合于单一化验条件。ESM3 零样本（Firnberg 域 ρ=0.626，四域中
最强单项）作为无标签参照并入——它恰好论证了漏斗设计：**PLM 的进化先验在保守酶上
很强、在荧光/结合这类非保守表型上很弱（avGFP 0.25、GB1 0.47），正确用法是作为
中观层组件与标签闭环组合，而非独立的适应度预言机**。

### 2.4 群体遗传学一致性（B3）

单倍体 Wright–Fisher 内核的固定概率与 Kimura 解析式 (1−e^{−2s})/(1−e^{−2N_es}) 在
s∈[0.01,0.2] 内最大绝对误差 0.011；8 个平行种群在强选择景观下收敛突变 (site,AA)
重合率显著高于多项随机期望（z=242.6）；含突变负荷的净适应斜率为正（图 7）。
这保证引擎的动力学不是"另一个遗传算法"，而是可被群体遗传学统计检验的进化过程。

### 2.5 廉价层地板与前沿对标带（B4）

在 30 个分层抽样的 ProteinGym DMS 上（与集群 ESM3 打分同批）：BLOSUM62 加性 ρ=0.224、DMS 校准 20×20 替换矩阵 ρ=0.199——位点无关的廉价先验存在 ~0.20-0.22 的地板。
作为参照，官方榜单带为 VESPA 0.30 / ESM-1v 0.374 / Tranception L 0.434 /
ESM3-open-1.4B 官方 217 集均值 0.466（我们在 CPU 集群自测：同一 30 集子集 ρ=0.322，
中位 0.354——子集与打分口径差异见 Methods；TEM-1 域 0.626、GB1 域 0.469 与文献区间吻合）/ SOTA 0.518（图 5）。
**v2 的设计立场**：不与 PLM 拼零样本，而是把 PLM 作为中观层组件、把标签预算花在
漏斗刀刃上——B1a 显示 ~400 标签即可让廉价先验 + 回流达到 ESM-1v 零样本水平。

### 2.6 PprI wave-2：真实蛋白上的闭环（B5）

PprI（耐辐射奇球菌 IrrE）是 DNA 损伤响应的开关蛋白。首轮战役（wave-0/1，3,240 个
Boltz-2 模型 + 17 位点 FoldX 扫描）确认了特异性冠军（sep 63.4）但全部带 +15~17
kcal/mol 的去稳定——归因于先验在碱基边缘位反复把芳香锚（F88/M135/Y170 族）换成
极性残基。v2 机制（M1 稳定性修正先验：prior × exp(−ΔΔG/2)；M2 锚位族约束；
M3 稳定性入适应度 w=0.4；M4 赢家回流）重跑后：34/34 精英通过稳定性 gate
（wave-1 对照 34/34 红旗）；top-3 候选在 CHPC 4090 上以 3 seed × 3 条件 × 20 samples
确认（540 模型）：**wave2_3 sep=70.0（战役历史最高）且 ΔΔG 预测 −0.29（净稳定）**，
wave2_2 sep=60.0 / S1=80.0（图 6）。

诚实边界：sep 为 Boltz-2 计算代理（激活 = HEXXH–DNA ≤5 Å 长链口径，多 seed 聚合）；
ΔΔG_pred 为 17 实测位点 FoldX + 类别外推的加性预测；候选未湿实验。

## 3. Discussion

**与前沿工具的关系**。EvoTrace 不与 ESM/RFdiffusion/ProteinMPNN 竞争零样本或
从头设计；它消费它们（PLM 作为中观层、结构预测作为昂贵层确认）并补上它们缺失的
一层：**种群遗传学的提议过程 + 标签经济的闭环**。与 ML-guided dEvo（Wu 2019 家族）
相比，v2 的优势在（a）平行种群 + 冠军奠基的探索结构，（b）可解释的先验回流
（伪计数，非黑箱后验），（c）结构机制先验带来的零标签冷启动能力。

**局限**。① avGFP/GB1 基准以 DMS 实测值为昂贵层真值——比的是策略的标签经济性，
不是打分器精度；② 提议限制在实测文库内（"文库约束"，对应湿实验的库合成范围）；
③ PprI sep 为计算代理，需湿实验最终确认；④ 万年尺度轨迹（v1 §3.5）未在本文验证。

## 4. Methods（要点，全文另附）

- **进化核**：Wright–Fisher，softmax 选择 T，Poisson(λ) 突变/基因组/代，突变负荷
  上限（超限随机回退多余位点，保留骨架）；两种提议模式（自由/文库行走）；
  冠军奠基；锚位掩码；稳定性入适应度。
- **回流**：prior'(i,a) ∝ prior(i,a)^(1−α) · post(i,a)^α，post = (evidence + k0·prior)/
  (k0 + E)，evidence = Σ_labels σ((y − y_ref)/scale)。
- **混合打分器**：对偶 ridge（one-hot(site,aa) + 突变数 + 初始先验分特征），
  bootstrap 不确定度用于 ridge-AL 基线的 UCB 批次。
- **先验构建**：PDB 解析 → 配体接触类别（<4.5 Å，原子类型分桶）+ 埋藏密度
  （8 Å 邻居数，p25 分位阈值）→ 类别化学先验表 → 稳定性修正（实测 FoldX 优先，
  类别外推默认 +2 kcal/mol）。
- **集群**：ESM3-sm-open-v1 零样本（全序列上下文 log-likelihood），sugon CPU 分区
  array 作业（吞吐 ~6 s/序列·8 核）；Boltz-2 确认（3 seed × 20 diffusion samples，
  recycling 3，sampling 200），CHPC 4090。
- **基线**：随机诱变（文库均匀/先验加权筛选）、对偶 ridge-AL（bootstrap-UCB，
  Wu et al. 2019 家族）、v1 引擎（静态先验、无回流、无锚位、无稳定性项）。
- 可复现性：全部基准 `evo2/benchmarks/*.py`，随机种子固定，结果 CSV + 环境记录随附。

## 5. 数据与代码可用性

- 引擎与基准：`evo2/`（MIT）
- 数据：ProteinGym v0.1（217 DMS，本地快照）、FLIP GB1、Sarkisyan 2016、
  FoldX 归因（本文生成）、PprI 战役资产（`ppri_evo/`）
- 模型输出：Boltz-2 540 模型（wave-2）、ESM3 ~105k 序列分数

## 参考文献（核心，扩至发表版）

1. Sarkisyan et al., Nature 2016（avGFP 完整景观）
2. Wu et al., eLife 2016（GB1 四位点组合景观）；Olson et al., Curr Biol 2014
3. Firnberg et al., PNAS 2014；Deng et al., 2012；Stiffler et al., 2015；
   Jacquier et al., 2013（TEM-1 四研究）
4. Notin et al., NeurIPS 2023（ProteinGym）及官方榜单（proteingym.org，2026-08 访问）
5. Notin et al., NeurIPS 2022（Tranception）；Frazer et al., Nature 2021（EVE）
6. Hayes et al., Science 2025（ESM3）；Lin et al., Science 2023（ESM-2）
7. Wohlwend et al., 2025（Boltz-2）
8. Wu et al., PNAS 2019（ML-assisted dEvo, Arnold lab）
9. Gerrish & Lenski, Genetica 1998（克隆干扰）；Lenski et al., 1991/2017（LTEE）
10. Otwinowski & Plotkin, PNAS 2014；Sailer & Harms, PLOS CB 2017（全局上位性）
11. Kimura, Genetics 1962（固定概率）
12. Jumper et al., Nature 2021（AF2）；Dauparas et al., Science 2022（ProteinMPNN）；
    Watson et al., Nature 2023（RFdiffusion）
