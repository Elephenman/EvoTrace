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

### 1.1 真实数据验证（PprI，8SLN + UniRef MSA 189 条）

合成验证之外，**真实数据同样支持该结论**（`b6_evoprior_demo.py` 输出）：

| 位点 M135（seq_idx 113，PDB 编号 135，offset=22） | top 偏好 | 疏水 (M/F/I/L/V) 合计 |
|---|---|---|
| `p_chem` 手工化学规则 | N=0.102, Q=0.091, S=0.091, **H=0.081**, T=0.081 | **0.132** |
| `p_evo` 数据驱动（同源 PSSM） | **M=0.464, L=0.214, I=0.137**, A=0.048, V=0.042 | **0.863** |
| `p_final` 融合（α=0.05，CV 学习） | 同 `p_evo` | **0.863** |

化学先验把该埋藏位推向极性残基（H 进入 top5，正是"M135H"的规则来源），
而同源 MSA 中该位 86.3% 为疏水。融合后盲点被完全纠正——这是
"规则驱动 vs 数据驱动"最直接的定量证据。

> ⚠️ 该结论依赖于 MSA 正确解析。此前 `read_a3m` 有一个静默 bug：把对齐缺口
> `'-'` 当作插入列一并删除，导致 189 行同源序列全部因长度不匹配被跳过，
> PSSM 退化为"query one-hot + 伪计数"（平均熵 2.978，几乎等于均匀分布
> 2.996），上表证据会完全消失。修复见 §4.0，回归测试见
> `test_read_a3m_preserves_gaps`。

## 2. 架构

```
   同源 MSA (a3m)                        蛋白序列
        │ read_a3m()                         │
        │ （保留 '-'，剔除小写插入列 §4.0）    │ 集群 ESM3-sm-open-v1
        ▼ msa_to_pssm()                      ▼ （两段式，§4.1）
   PSSM [L,20]  ←—— 标签/真值            嵌入 [L,1536]
        │                                    │ PCA → 32 维（§4.2）
        │                                    ▼
        │                              EvoPriorMLP（两层，纯 numpy）
        │                                    │ 留出 5 折 CV 预测
        └────────►  p_evo [L,20]  ◄──────────┘
                    （默认以 PSSM 为准，仅低覆盖列用 MLP 补全 —— §5.1）
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
# PprI 实测命令：PDB 链 A 残基 22-275，故 --seq_offset 22
python evo2/benchmarks/b6_evoprior_demo.py \
    --seq  ppri_evo/inputs/wt_254.fasta \
    --msa  ppri_evo/.../uniref.a3m \
    --pdb  ppri_evo/inputs/8SLN.pdb \
    --seq_offset 22 --chain A \
    --embed_npz evo2/cluster/payload/ppri_wt_esm3.npz \
    --pca_dim 32 --p_evo_source auto \
    --out out/evoprior_esm3
```

输出 `evoprior_report.txt` 含：PCA/嵌入来源、三基线留出 CE 诚实性评估、
PSSM 平均熵、MSA 列深度与低覆盖列数、α 交叉验证曲线、最大分歧位点诊断；
`evoprior_priors.csv` 含每位点 `p_chem` / `p_evo` / `p_final` 三套 20 维分布。

### 4.0 MSA 解析：gap 必须保留（重要 bug 修复）

`read_a3m` 曾用 `c.isupper()` 过滤字符，把对齐缺口 `'-'` 与小写插入列一并删除。
后果：同源序列被"压缩"、列整体错位、长度不再等于 query → `msa_to_pssm` 因
`len(s) != L` **静默跳过几乎所有行**（seen=1，只剩 query 自身）→ PSSM 退化为
"query one-hot + 伪计数"。

| | 修复前 | 修复后 |
|---|---|---|
| PSSM 平均熵 | 2.9785（≈均匀 2.9957） | **1.8963** |
| 保守位（maxP>0.5） | 0 / 254 | **109 / 254** |
| M135 疏水概率 | 无信号 | **0.863** |

现已修复为"只剔除小写插入列、保留 `'-'`"，并在 `msa_to_pssm` 中对"过半行被
跳过"发出 `RuntimeWarning`。回归测试：`test_read_a3m_preserves_gaps`。

### 4.1 真实 PLM 嵌入：集群 ESM3 两段式（已打通）

本地常因网络限制无法下载 PLM 权重（HuggingFace 不可达），而 CPU 集群
（`u22607007@10.205.1.3:10022`）的 `esm3` conda 环境已具备 ESM3-sm-open-v1
权重。因此采用**两段式**：

```
集群（CPU，esm3 env）                    本地（无需 torch）
  cluster/esm3_embed_cluster.py    →     engine/evoprior.py
  --fasta seqs.fa --out emb.npz          load_npz_embeddings(emb.npz)
  输出 [L, 1536] per-residue 嵌入         → PCA 降维 → MLP / PSSM 融合
```

```bash
# ① 集群：提取嵌入（PprI 254aa 约 20s，含模型加载 14s）
scp esm3_embed_cluster.py seqs.fasta u22607007@10.205.1.3:~/evo2_esm/evoprior/
ssh -p 10022 u22607007@10.205.1.3 "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate esm3 &&
  cd ~/evo2_esm/evoprior && python esm3_embed_cluster.py --fasta seqs.fasta --out emb.npz"
scp -P 10022 u22607007@10.205.1.3:~/evo2_esm/evoprior/emb.npz ./evo2/cluster/payload/

# ② 本地：训练 EvoPrior
python evo2/benchmarks/b6_evoprior_demo.py --seq ... --msa ... --pdb ... \
    --seq_offset 22 --embed_npz evo2/cluster/payload/emb.npz --pca_dim 32
```

**集群侧的坑**：`esm` 包的 `data_root()` 走 `snapshot_download()`，而集群上
HF 缓存只有 refs、缺 blobs 与 metadata（网络也不可达），必然抛
`LocalEntryNotFoundError`。脚本中的 `install_offline_data_root()` 在导入
`esm.pretrained` **之前**把 `data_root` 打补丁指向本地权重目录
`~/models/esm3-sm-open-v1`（权重与 tokenizer CSV 都在），绕开 HF 缓存。
（另有 `INFRA_PROVIDER` 环境变量短路，但那会让路径变为相对路径，不适用。）

### 4.2 嵌入维度的处理

ESM3-sm-open-v1 嵌入为 **1536 维**，而单条蛋白只有 ~254 个位点，`D ≫ L`
直接训练必然过拟合。故默认 `pca_dim=32`（解释方差约 0.48）。

## 5. 方法学贡献（论文升级点）

| 维度 | v2（规则驱动） | v3 + EvoPrior（数据驱动） |
|---|---|---|
| 先验来源 | 手工化学规则 `p_chem` | `p_final = α·p_chem + (1−α)·p_evo` |
| 135 位 | 盲点（M135H 错误） | 数据驱动自动纠正（疏水偏好） |
| 归因 | "揭示手工规则盲点" | "用 DL 自动识别并规避盲点" |
| 叙事 | 弱点 | 方法学创新证据 |

## 5.1 实测结论：`p_evo` 该用 PSSM 还是 MLP？（诚实评估）

用户原始设想是"训练 MLP 从 PLM 嵌入预测 PSSM"。我们在 PprI 上做了严格的
**留出位点 5 折交叉验证**（254 位点，MSA 189 条，ESM3 嵌入 PCA→32 维），
并设置了两个**不含嵌入**的基线：

| 方法 | 留出交叉熵 CE | 说明 |
|---|---|---|
| uniform（1/20 均匀） | 2.9957 | 下界参照（CE 恒为 ln20） |
| composition（全局氨基酸组成） | 2.8537 | 弱基线 |
| MLP（ESM3 嵌入，wd=3e-2） | 2.5173 | 纯嵌入 |
| MLP（WT one-hot + ESM3，wd=1e-2） | 2.4473 | 嵌入 + 身份 |
| MLP（WT one-hot，wd=1e-4） | 2.3057 | 纯身份 |
| **wt_cond 查表（无嵌入）** | **2.2242** | **最强**：按 WT 残基类型取平均 PSSM |
| PSSM 自身平均熵（记忆下界） | 1.8963 | 非预测值，仅作参照 |

结论（三条，均需如实写入论文）：

1. **ESM3 嵌入确实携带位点信息**：显著优于 uniform 与 composition
   （ΔCE ≈ 0.34–0.48），说明 PLM 编码了真实的位点环境信号。
2. **但单蛋白样本量下，嵌入无法超越"按 WT 残基类型取平均"**：PSSM 的位点间
   变异主要由"该位点当前是什么氨基酸"解释。MLP 最好 2.3057，仍差于查表
   2.2242。
3. **一个易犯的分析陷阱**：在强正则（`w_decay=1e-1`）下会观察到
   "WT+嵌入 优于 WT-only"（ΔCE=+0.108，t=+9.87，看起来很显著）；但放宽正则
   让两者各自取最优超参后，结论**反转为 t=−7.15**。前者是欠拟合区的假象。
   → 教训：**小样本下必须先扫正则化强度再比较特征，否则会得到假阳性结论**。

因此 `build_evoprior(p_evo_source=...)` 提供三种模式，默认 `auto`：

- `pssm`：直接用 PSSM 作 `p_evo`（有 MSA 覆盖时，观测值比任何模型估计可信）；
- `mlp`：纯 MLP 留出预测；
- `auto`（默认）：PSSM 列深度 ≥ `min_neff`（默认 30）的列用 PSSM，
  不足的列用 MLP 预测补全（PprI 实测仅 5/254 列需要补全）。

> **论文叙事建议**：EvoPrior 的核心贡献是"用同源进化证据（PSSM）自动纠正手工
> 规则盲点"，这一点已被真实数据坐实（§1.1）。PLM 嵌入的增量价值则需要
> **多蛋白联合训练**才能兑现——单蛋白 254 个位点不足以训练出超越 WT 查表的
> 预测器。这不是缺陷，而是下一步的方法学空间（见 §6）。

## 6. 可优化方向（留给本分支继续）

- [x] **真实 PLM 嵌入接入（CPU 集群 ESM3 两段式）— 已实现（§4.1）**
- [x] **MSA 解析 gap bug 修复 + 回归测试 — 已修复（§4.0）**
- [x] **α 的逐位点自适应（而非全局常数）— 已实现（§6③）**
- [x] **α 交叉验证曲线修正** — 原先只在验证折上评估"训练集选出的那个 α"，
      导致其余 α 桶保持初值 0（伪最优曲线）；现改为每个候选 α 都在验证折评估。
- [ ] **多蛋白联合训练（PprI 同源家族）** ← 兑现 PLM 增量的关键下一步
- [ ] MLP → 图神经网络（残基为节点，距离边，显式 3D 上下文）
- [ ] MSA 序列加权（去冗余 / position-based weighting）提升 PSSM 标签质量
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
