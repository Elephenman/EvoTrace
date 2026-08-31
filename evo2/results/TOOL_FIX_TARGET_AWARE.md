# EvoTrace 工具修复：DNA 条件门控景观（让工具能朝靶标进化）

> 状态：v1.1 工具硬伤已修复并原型验证通过。待确认后提交主分支。
> 靶标：24nt S1_G17 `TCATGAGCAGTTTTTTGTTTTTTT`（pos17=G / pos23=T 读头）

## 1. 问题诊断（为什么 v1.1 "不行"）

v1.1 的 PprI 景观 `surrogate_ppri_v3`（ESM3 代理）是**零样本、与 DNA 序列无关**的通用适应度 z-score 预测器：

- 它只预测"蛋白突变后的通用稳定性/表达"，景观里**完全没有 DNA 维度**。
- 对它做"朝靶标优化"在数学上等价于"最大化天然样 z" → 只会把设计突变**洗回 WT**，甚至把**结合必需锚点洗掉**。
- 实测（evolve_six_wetlab）：HQL2 的 7 个机制突变被全洗回 WT，Δz +0.794 是假象。

证据（evolve_target_aware 对比，旧纯 v3 vs 新 DNA-aware，WF 从 WT 演化 80 代）：

| 终末种群机制保留率 | 旧（纯 v3） | 新（DNA-aware） |
|---|---:|---:|
| 锚点 R85/R207/R267 保留 | **0%**（全改掉→结合归零） | **100%** |
| 双锁 R253/Y217/M255 正电 | 35%（随机） | **100%** |
| 读头 F88 芳香率 | 28%（随机） | **100%** |

旧景观 top（z=0.678）把 R85→H、R207→F、R267→F 全改掉；而 R85A/R207A/R267A 三突变体验证过**完全丧失 ssDNA 结合**——旧景观优化出的是生物学毁灭性变体。

## 2. 修法：DNA 条件门控景观

组合景观，把"保结构"与"靶标匹配"解耦：

```
fitness = w_base · z(ESM3 稳定性/表达)  +  w_gate · gate(靶标匹配机制)
```

`gate` 由 PprI 已知机制编码（**数据驱动**，源自 `priors.csv` 的 `dominant_class`）：

- **读头（F88 系，base_edge）**：对靶标 readhead 碱基 G 的 π-堆叠/氢键偏好 − 对非靶 T 偏好 = 特异性
- **双锁（R253/Y217/M255，base_edge，nearest_nt23）**：正电/极性残基保留或增强 DNA 接触
- **锚点（R85/R207/R267，phosphate/base_edge）**：必须为 R 以捕获 ssDNA 磷酸骨架，改掉则惩罚

机制热点用 `pdb_resi` 常量（已与 priors.csv 校验：F88=seq_idx66、R253=231、Y217=195、M255=233、R85=63、R207=185、R267=245 均在 53 可变空间内；HEXXH/K67/R232 不在→冻结）。

## 3. 实现

| 文件 | 角色 |
|---|---|
| `evo2/esm3/ppri_dna_aware.py` | `DnaAwareLandscape` 类 + `make_dna_aware` 工厂。接口对齐搜索内核（L/wt_idx/sites/evaluate/n_mutations/enforce_max_mut/max_mut） |
| `evo2/benchmarks/b7_three_way.py` | 注册 `dna_aware_ppri` 为正式 oracle（main 内 REBUILD，surrogate 成功时激活） |
| `evo2/benchmarks/evolve_target_aware.py` | OLD vs NEW WF 演化对比验证 |
| `evo2/benchmarks/verify_b7_dna_aware.py` | b7 接口对齐 + 机制惩罚验证 |
| `evo2/results/evolve_target_aware.csv` | 验证数据 |

## 4. 验证结果

- b7 语法 OK，`dna_aware_ppri` 注册生效，搜索内核接口对齐。
- 机制惩罚正确：WT=+0.162，F88Y=+0.113（读头保留），R85A=−0.110（锚点破坏被罚 ✓）。
- WF 演化：新景观终末种群锚点/双锁/读头机制保留率均 100%，旧景观机制全毁。
- **结论**：工具现在具备 DNA 维度，能朝靶标进化，不再洗回/毁机制。

## 5. 诚实局限与下一步

1. **gate 是机制启发式，非学习到的**。权重 w_base/w_gate=0.5 与匹配规则需后续用 **Boltz-2 判读标签**校准（job 225685 在跑）。
2. **靶标特异性（靶 vs 非靶判别）终验仍需 Boltz**——gate 保证"保留机制"，但"双锁=靶标特异激活"的最终排序由 Boltz 给出。
3. **当前为原型**（CPU WF pop40×gen80）。接入 b7 后可换 PPO/CEM 做更大规模靶标条件搜索。
4. 下一步建议：用 Boltz 判读数据集（靶/非靶分离分数）训练一个**小样本条件代理**（蛋白突变×DNA→特异性），替代解析 gate，使工具从"机制编码"升级到"数据驱动靶标景观"。

## 6. 对湿实验清单的影响

- 六候选（s13_c1/TrackF_r1/WT/HQL2/RD_POS/RDP2）的原始设计仍是靶标最优解（由完整战役 + Boltz 设计）。
- 新工具的价值：后续**增量轮次**可用 `dna_aware_ppri` 在保留机制前提下，搜索比六候选更稳定/更高表达的变体，而非把设计洗回 WT。
- 等 Boltz job 225685 判读后，用 `dna_aware_ppri` 对六候选做"保机制精修"，产出 v2 湿实验候选。
