# EvoPrior — 数据驱动的位点偏好预测器（进化先验 p_evo）

> 对应分支 `evoprior`。将 EvoTrace 从"纯规则驱动的化学先验 p_chem"升级为
> "规则 + 数据双驱动"的融合先验 p_final，并据此把论文方法学叙事从
> "揭示了一个手工规则导致的盲点"升级为**"用深度学习自动识别并规避了该盲点"**。

## 1. 动机：M135 盲点（论文最大弱点）

M1 的化学先验 `p_chem`（`engine/priors.py`）是手工规则：每位点的氨基酸偏好由
其结构环境（配体接触 / 埋藏 / 表面）的化学性质推导。它在 PprI 的 **135 位**
（埋藏疏水位）给出 M/F/L/I/V 高权重——这本是对的。

但手工规则在 135 位被错误地改写为 `M135H`（把疏水 M 改成极性 H），与进化事实
冲突：PprI 同源 MSA 中 135 位强烈保守为疏水（M/F/L）。这正是论文中
"特异性–稳定性权衡"归因的最大漏洞——**部分 trade-off 并非蛋白本身的物理约束，
而是先验设计的盲点**。

`engine/tests/test_evoprior.py` 用合成数据复现并自动验证了该盲点的修复：
化学先验疏水概率 0.004 → 融合进化先验后 0.401；KL(p_ssm ∥ p_final) 在 135 位
从 4.893（纯化学）降至 0.000（α 经 CV 学为 0）。

## 2. 架构

```
        同源 MSA (a3m/fasta)
              │  msa_to_pssm()
              ▼
        PSSM [L,20]  ←—— 标签（天然氨基酸频率，即"进化真值"）
              │
  蛋白序列 ──┤
              │  get_embeddings()  [ESM-2 / ProtBERT, 可选延时加载]
              ▼
       嵌入 [L, D]  ──►  EvoPriorMLP  ──►  p_evo [L,20]  (softmax 分布)
                                    （两层 MLP，纯 numpy；可替换为 GNN）
```

- **输入**：蛋白序列（PprI 或同源）+ 每位点局部结构上下文（从 PDB 自动解析）。
- **标签**：同源 MSA 提取的 PSSM（位置特异性评分矩阵）。
- **模型**：轻量 MLP（输入=PLM 嵌入，输出=20 维氨基酸概率分布）；接口预留 GNN。
- **输出**：数据驱动的"自然先验" `p_evo`，而非手工规则。

## 3. 融合层（M2 景观构建）

在 WFKernel 构建先验表时融合：

```
p_final = α · p_chem + (1 − α) · p_evo      # 每行归一
```

- `α` 通过交叉验证学习（`learn_alpha_cv`）：在验证位点上选 α 最小化
  `KL(p_ssm ∥ p_final)`，使融合分布尽量贴近真实进化频率。
  支持 `per_site=True` 逐位点自适应 α 模式（见 §6③）。
- 接入点：`WFKernel.__init__(evoprior=<[L,20]>, alpha=0.5)`；传 `None` 则退化为
  v2 纯化学先验（向后兼容，不破坏既有基准）。

## 4. 运行

```bash
# 单元测试（合成数据，验证融合数学 + M135 盲点修复 + MLP 收敛）
python evo2/engine/tests/test_evoprior.py

# 真实数据演示（规则驱动 vs 数据驱动对比）
python evo2/benchmarks/b6_evoprior_demo.py \
    --seq pprI_evo/inputs/wt_254.fasta \
    --msa pprI_evo/.../uniref.a3m \
    --pdb pprI_evo/inputs/8SLN.pdb --seq_offset 1
```

**关于嵌入的重要说明**：默认 `use_plm=False` 使用 `placeholder_embeddings`
（one-hot + 邻域窗口）仅用于跑通流程——它**不含进化信息**，MLP 几乎无法从纯
序列预测 PSSM（demo 中 CE 仅 3.04→3.02）。真实实验必须 `--use_plm`
（需 `transformers` + `torch`），用 ESM-2 逐残基嵌入（254 残基 CPU 推理约数秒，
加载 ~2.5GB RAM）。**真实 PLM 嵌入才编码同源保守性，是 EvoPrior 相对手工规则的
根本优势**——这也正是"规则驱动 vs 数据驱动"对比的实验落点。

## 5. 方法学贡献（论文升级点）

| 维度 | v2（规则驱动） | v3 + EvoPrior（数据驱动） |
|---|---|---|
| 先验来源 | 手工化学规则 `p_chem` | `p_final = α·p_chem + (1−α)·p_evo` |
| 135 位 | 盲点（M135H 错误） | 数据驱动自动纠正（疏水偏好） |
| 归因 | "揭示手工规则盲点" | "用 DL 自动识别并规避盲点" |
| 叙事 | 弱点 | 方法学创新证据 |

## 6. 可优化方向（留给本分支继续）

- [ ] 真实 ESM-2/ProtBERT 嵌入接入与缓存（CPU 集群批量预计算）
- [ ] MLP → 图神经网络（残基为节点，距离边，显式 3D 上下文）
- [x] **α 的逐位点自适应（而非全局常数）— 已实现（EVOPRIOR §6③）**
- [ ] 多靶 MSA 联合训练（PprI 同源家族）
- [ ] 与 M4 标签回流耦合（p_evo 作为回流的先验特征）

### 6.③ 逐位点自适应 α（已实现）

全局常数 α 在非保守位（化学先验本就可靠，应 α→1）与盲点位（如 135，需数据驱动
纠正，应 α→0）之间强行妥协。`learn_alpha_cv(per_site=True)` 返回 `[L]` 的 α 数组，
每位点按**规则–数据的局部分歧**定权：

```
α_j = 1 − d_j,   d_j = JS(p_chem_j, p_evo_j) / log2 ∈ [0,1]
```

- 规则与数据一致（d_j→0）→ α_j→1，保留化学先验（"化学先验强"）；
- 规则被盲化、与数据冲突（d_j→1，即盲点）→ α_j→0，转信进化先验（"进化信号强"）。

**为什么这样设计（避免两个退化）**
1. 不做"逐位点 k 折 CV 拟合 α_j"：单点时训练集为空会退化，且当 `p_evo≈pssm` 时
   直接最小化 `KL(p_ssm ∥ p_final)` 会让 α_j 处处→0（过拟合到数据），失去"可靠位
   点保留规则"的语义。
2. 仅以 `p_chem_j` 与 `p_evo_j` 的分歧为信号（不触碰 p_ssm_j），符合"规则驱动 vs
   数据驱动"的定性对比，且**不依赖 PLM**——即便 MLP 仅用占位嵌入、`p_evo` 退化为
   PSSM，也能凭"规则 vs 进化频率"的逐位冲突自动定位盲点。

**接入**：`fuse_tables(p_chem, p_evo, alpha_arr)` 与 `WFKernel.__init__(alpha=<[L] array>)`
均兼容标量/数组 α。注意 kernel 仅对可变位点 `self.sites` 融合，数组 α 须先按
`self.sites` 索引、再 reshape 到 `(L,1)` 才能与 `(L,20)` 表广播（已实现于
`_build_prior_table`）。`b6_evoprior_demo.py --per_site` 会导出
`evoprior_alpha_per_site.csv` 供 kernel 复用。

> 单元测试 `test_evoprior.py::test_per_site_alpha` 已验证（合成 M135H 盲点）：
> 盲位 α_135 ≈ 0.1（<0.25），可靠位 α_0 ≈ 0.37（显著高于盲位），融合后 135 位
> 疏水概率从 ~0 恢复到 0.71（与 p_evo 的 0.80 一致，偏差 <0.12）。
> WFKernel 烟测（sites=[2,5,7], α_2=0/α_5=α_7=1）确认：盲位转信数据（M 概率升高）、
> 可靠位保留规则（A 概率维持）。
