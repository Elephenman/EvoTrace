# EvoTrace v2 设计方案 — PPrI 战役驱动的引擎重构（2026-08-30）

> 定位：把 PprI（PprI/IrrE 辐射抗性蛋白）首次实战暴露的四个系统性问题，转化为引擎的一般性
> 机制升级；在**完整实测景观**（avGFP 51,714 变体、GB1 149,361 变体、TEM-1 4,783 单突变）
> 上做可复现基准，与**国际前沿工具**（ESM3 零样本自测 + ProteinGym/FLIP 公开榜单）对标；
> 以 PprI wave-2 + CHPC Boltz-2 确认收口真实蛋白战役。目标形态：可发顶刊的
> 方法论文（新引擎 + 四层验证 + 真实战役）。

---

## 1. PPrI 战役教训 → v2 机制映射（本文档的核心逻辑）

| # | PprI 实战暴露的问题 | 证据 | v2 机制 |
|---|---|---|---|
| L1 | **几何化学先验有系统性盲区**：碱基边缘位偏好极性氢键化学（N/Q/S/H 0.8-1.0），低估芳香 π-stack/疏水包裹（F/Y/W/M 0.3-0.5） | 17 位点 FoldX 扫描：135H +8.55、88Q +2.67（F88 π-stack→极性）、171N +1.12 kcal/mol | **机制 M1（稳定性修正先验）**：prior'' ∝ prior' × exp(−ΔΔG_site(i,aa)/τ)，τ=2 kcal/mol；ΔΔG 来源可插拔（FoldX 实测 / 类别外推） |
| L2 | **锚位反复被破坏**：进化在 88/135/171 把芳香锚换成极性 AA | wave-0/wave-1 精英均命中这三个位点 | **机制 M2（锚位约束）**：位点类 flag `anchor=True` 时，突变提议限制在结构化学族内（芳香族 F/Y/W/M；疏水族 I/L/V/M），族内自由、跨族禁止 |
| L3 | **稳定性只在事后门槛，不在适应度里**：精英漏斗放行了 +15~17 kcal/mol 的去稳定冠军 | e18/e3/w4 确认态全部 FoldX 红旗 | **机制 M3（稳定性入适应度）**：f = Σs(i,a) + w_stab·(−ΔΔĜ)，精英每 K 代回填实测/代理 ΔΔG |
| L4 | **标签只在代际之后批量回流，且蒸馏失败**（Phase-1 特征蒸馏 LOOCV ρ=0.06） | results/phase1_distiller.json | **机制 M4（贝叶斯标签回流=主动学习内核）**：正式化为 campaign 循环——廉价核进化 → 精英批 → 昂贵标签 → prior'(i,a) ∝ prior(i,a)^(1−α) × Laplace 后验^α（wave-1 机制的泛化 + 消融可关）；放弃黑箱蒸馏，保留可解释伪计数 |
| L5 | **廉价层与昂贵层分层有效**（确认门拦下了全部去稳定冠军），但分层无理论刻画 | 确认态排行 | **机制 M5（层级漏斗正式化）**：cheap（先验核，逐体）→ mid（PLM ΔLL，精英）→ expensive（结构/共折叠，候选批）三级接口 + 硬门槛（自然度/ΔΔG/催化保守），全部可配置 |

## 2. v2 引擎架构（A:\claudework\evo2）

```
evo2/
├── seqtools.py        # 序列/突变 IO（PDB 编号↔seq idx、突变串解析、FASTA）
├── priors.py          # 结构→位点类→化学先验（PDB 解析、配体接触、类别先验库、锚位标注）
│                      #   + 稳定性修正（M1）+ 锚位约束表（M2）
├── kernel.py          # 向量化 Wright-Fisher 核（M3/M5）：numpy 全向量化
│                      #   Ne×L 位点 int8 种群、查表适应度、softmax 选择、Poisson 突变、
│                      #   平行种群、谱系统计、克隆干扰记录、锚位掩码
├── reflux.py          # 贝叶斯标签回流（M4）：伪计数证据 + prior 幂混合 + 全局上位性层 g(·)
├── funnel.py          # 层级漏斗（M5）：三级打分接口 + 硬门槛 + 批次提议器（top-k+多样性）
├── landscape.py       # 基准用景观抽象：OracleLandscape（实测真值+预算标签）、ProxyLandscape
├── baselines.py       # 对标基线：随机诱变 / 贪婪 / ridge-AL(Wu 2019) / v1 引擎 / ESM3 零样本(外部)
├── metrics.py         # Spearman、标签经济曲线、终点保真、固定概率、平行进化 KS、克隆干扰
└── benchmarks/        # B1a/B1b/B2/B3/B4/B5 运行脚本 + 结果 CSV
```

**关键设计决策**（相对 v1 引擎 evolve.py）：
1. **种群向量化**：v1 是纯 Python 逐体循环（Ne=500×15 代×4 种群可跑但慢）；v2 种群 = `int8[Ne, L]`，
   适应度 = 选择系数查表 `sel[arange(L), geno].sum(1)`，突变/选择全 numpy——支持 Ne 10⁵ 级。
2. **基因型空间开放**：v1 硬编码 53 个 PprI 可突变位点；v2 由 priors.csv 通用 schema 驱动（任意蛋白）。
3. **可插拔分量**：适应度 = 先验选择系数 + w_stab·稳定性 + w_epi·全局上位性变换，全部开关化（消融实验即配置）。

## 3. 基准体系（对标国际前沿的证据链）

| 基准 | 数据 | 对标对象 | 指标 |
|---|---|---|---|
| **B1a avGFP 全景观** | Sarkisyan 2016（51,714 实测变体，完整局部景观）+ 1EMA 结构先验 | 随机诱变 / 贪婪 / ridge-AL / v1 引擎 / ESM3 零样本 | 标签经济曲线（真值适应度 vs 标签预算）、终点 top-16 真值、Spearman |
| **B1b GB1 四位点** | Wu 2016 eLife（149,361 变体）+ FLIP 官方 two-vs-rest 划分 | FLIP 公开榜单（ESM-2/1v、Tranception 等）+ ESM3 自测零样本 | test Spearman、top-16 真值 |
| **B2 TEM-1 表位** | Firnberg 2014（4,783 单突变，训练正交）+ Jacquier 2013（双突变，外推验证） | 加性模型基线；全局上位性层 g(·) 的增益 | 双突变 Spearman（加性 vs +g 层 vs ESM3） |
| **B3 群体遗传理论一致性** | 合成景观 | Kimura 固定概率解析式、平行进化多项检验 | 经验 vs 解析 P_fix（KS）、平行度 p 值 |
| **B4 ProteinGym 217** | 本地 217 个 DMS（270 万突变体） | 公开榜单参考带（参考成绩 CSV）+ ESM3 零样本自测（CPU 集群） | 平均 Spearman（物理地板 + ESM3 vs 榜单带） |
| **B5 PprI wave-2 实战** | 8SLN 结构先验 + 17 位点 FoldX 实测 ΔΔG + 已确认赢家证据 | wave-1（无 M1/M2 的上一代）| wave-2 精英的预测 ΔΔG 分布、锚位违规数、Boltz-2 确认 sep（CHPC） |

**科学诚实边界**（写入论文）：① 基准中"昂贵标签"= DMS 实测值本身（oracle 模型独立于方法，
比的是**进化策略的标签经济性**而非打分器精度）；② avGFP/GB1 提议限制在实测基因型集内
（"文库约束"，湿实验对应饱和突变库）；③ PprI wave-2 的 Boltz-2 确认为计算代理，未湿实验。

## 4. 集群分工

| 资源 | 用途 |
|---|---|
| 本地 CPU（Python 3.12 + numpy） | evo2 内核、全部进化模拟、统计分析、图表 |
| CPU 集群 10.205.1.3（sugon 分区，21 节点） | ESM3-sm-open-v1 零样本打分：avGFP 评测面板、GB1 two-vs-rest 测试子样、TEM-1、ProteinGym 217 子集 |
| GPU 集群 CHPC 10.202.94.52（4090 分区） | Boltz-2 wave-2 候选确认（3 候选 × 3 条件 × 3 seed × 20 samples ≈ 540 模型） |

## 5. 执行顺序（长任务先行）

1. evo2 核心包 + 单元自检（本地，~1h）
2. B1a/B1b/B2/B3 全量本地运行（数小时内）
3. ESM3 打分载荷上 CPU 集群（并行跑，等结果期间继续本地工作）
4. B5 wave-2 先验修正 + 引擎运行 + CHPC Boltz 提交
5. 图表 + 论文草稿 + 技术方案 v2 + 交付报告
