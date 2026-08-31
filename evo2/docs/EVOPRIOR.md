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
化学先验疏水概率 0.140 → 融合进化先验后 0.469；KL(p_ssm ∥ p_final) 在 135 位
从 1.346（纯化学）降至 0.000（α 经 CV 学为 0）。

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
- [ ] α 的逐位点自适应（而非全局常数）
- [ ] 多靶 MSA 联合训练（PprI 同源家族）
- [ ] 与 M4 标签回流耦合（p_evo 作为回流的先验特征）
