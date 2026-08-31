# EvoTrace · 基于蛋白质大模型的数字定向进化模拟器

> 一个把"定向进化实验"搬到计算端的开源引擎：用蛋白质语言模型（PLM）嵌入与结构化学先验，
> 在数字景观上做 Wright–Fisher 进化模拟，并以主动学习漏斗高效分配昂贵的实验/验证标签预算。

## 核心架构（三映射）

| 层 | 模块 | 职责 |
|---|---|---|
| **M1 环境编译** | `engine/priors.py` | PDB → 位点类别（配体接触/埋藏/表面/发色团核）→ 化学先验；稳定性修正先验 `prior'' ∝ prior'·exp(−ΔΔG/τ)` |
| **M2 景观组装** | `engine/kernel.py` | 向量化 Wright–Fisher 核；锚位掩码、稳定性入适应度、平行种群、固定事件 |
| **M3 进化动力学** | `engine/reflux.py` + `funnel.py` | 贝叶斯标签回流（伪计数证据 + 先验幂混合）+ 三级漏斗（先验核→PLM→Boltz 确认）|

## 五大机制（v2，来自 PprI 辐射抗性蛋白战役实战）

- **L1** 几何化学先验盲区 → 稳定性修正先验（M1）
- **L2** 锚位反复被破坏 → 锚位族内替换约束（M2）
- **L3** 稳定性只在事后门槛 → 稳定性入适应度（M2/M3）
- **L4** 标签只事后回流、黑箱蒸馏失败 → 贝叶斯标签回流（M3）
- **L5** 分层漏斗无正式化 → 三级漏斗 + 硬门槛（M3）

## 目录结构

```
evo2/
├── engine/        # 模拟内核（seqtools/priors/kernel/reflux/funnel/baselines/metrics）
├── benchmarks/    # b1a avGFP / b1b GB1 / b2 TEM-1 / b3 理论 / b4 ProteinGym / b5 PprI
├── cluster/       # CPU 集群(ESM3 打分) / GPU 集群(Boltz-2 确认) 工具
├── docs/          # DESIGN_V2.md / FINAL_REPORT.md
├── figures/       # 基准结果图
├── paper/         # EVOTRACE_MANUSCRIPT.md
└── results/       # 基准 CSV / 汇总 JSON / 日志（esm3 打分中间产物已 gitignore）
```

## 基准结果（对标国际前沿）

| 基准 | 关键结果 |
|---|---|
| B1a avGFP | v2 ρ=0.512±0.017（384 标签）超 ESM-1v 0.374、ESM3 1.4B 零样本 0.251 |
| B1b GB1 | test ρ=0.575→0.707，top16 3.58/4.40 |
| B2 TEM-1 | 跨条件迁移 16/16 全胜 ridge 基线 |
| B3 理论 | 固定概率 vs Kimura 误差 ≤0.011 |
| B4 ProteinGym | 廉价层地板 ρ≈0.20-0.22（vs 前沿带 0.47-0.52） |
| B5 PprI | wave-2：34/34 精英过稳定性 gate；sep=70.0 历史最高，ΔΔG 净稳定 |

## 环境

- Python 3.12 + numpy（全向量化内核，本地 CPU 即可跑全部模拟/统计/图表）
- CPU 集群（ESM3 零样本打分）/ GPU 集群（CHPC 4090，Boltz-2 确认）
- 运行：`python evo2/benchmarks/b1a_avgfp.py` 等

## 许可

待定（投稿前默认私有可见，后续可转为开源协议）。
