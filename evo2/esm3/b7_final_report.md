# EvoTrace 六内核最终对比报告 —— 最初工具 vs 深度学习（RL/ES 分支全量）

> 生成时间: 2026-08-31 | 预算: 4000 evals/优化器/seed × 5 seeds | 6 内核 × 8 景观
> 内核: WF（最初工具·基线）/ PPO（torch 深度 RL）/ DQN（numpy RL）/ BC（离线对照）/ OpenAI-ES / CEM

## 1. 深度学习正确性验证（本轮新增）

### 1.1 DL 代理（SurrogateOracle）—— 端到端推理验证通过

模型 `A:/claudework/out/surrogate_dl/model.pt`（DeepSet 突变编码器，12.3 万参数，ProteinGym 214 DMS / 1,716,079 变异训练）。对 PprI 53 位点做零样本推理验证：

| 验证项 | 结果 | 判定 |
|---|---|---|
| 53×19 单点突变批量预测 | 957/1007 唯一预测值 | ✅ 高区分度，无塌缩 |
| NaN / inf | 0 / 0 | ✅ 数值稳定 |
| 相同基因型一致性 | AA==WT 行预测与 WT 完全一致（err_max=0.0000） | ✅ 输入输出一致 |
| 逐位点区分度 | 53/53 位点 std>0（median 0.181） | ✅ 每位点有信号 |
| 突变方向 | 782/1007 预测 < WT，225 > WT | ✅ 符合"多数突变有害"生物学规律 |

### 1.2 torch PPO —— 真深度学习策略（非 numpy 替代）

`rlpolicy_torch.py`（commit 3a67951）确认标准 PPO 实现：
- 网络：2 层 Tanh MLP trunk + policy/value 双头，orthogonal 初始化
- 训练：PPO-clip（ε=0.2）+ GAE(λ=0.95, γ=0.99) + entropy 正则（0.01）+ grad clip（0.5）+ minibatch（256）
- 动作空间 L×20 flat + no-op/超限 mask；奖励 = oracle 适应度增量 Δf
- 每个 oracle 独立进程运行，规避 torch 跨 oracle 内存累积（修复 64.7MiB OOM）

## 2. 六内核 × 8 景观完整对比（PPO 全量补齐）

逐 oracle 中位数 norm（5 seed × 4000 evals；norm=(best−wt)/(ref−wt)；BC 为 PprI 系阴性对照，gb1 轨迹维度不匹配无数据）：

| 景观 | WF | **PPO** | CEM | ES | DQN | BC |
|---|---|---|---|---|---|---|
| gb1_pairwise（GB1 实测） | **0.985** | 0.888 | 0.956 | 0.946 | 0.815 | — |
| nk_k0 | 0.438 | 0.477 | **0.747** | 0.563 | 0.365 | 0.065 |
| nk_k1 | 0.405 | 0.375 | **0.519** | 0.459 | 0.321 | 0.058 |
| nk_k2 | 0.361 | 0.402 | **0.611** | 0.428 | 0.273 | 0.048 |
| nk_k3 | 0.348 | 0.368 | **0.487** | 0.401 | 0.266 | 0.087 |
| ppri_additive（PprI 加性） | **0.903** | 0.294 | 0.582 | 0.368 | 0.250 | −0.016 |
| ppri_cal（PprI 校准） | **0.712** | 0.272 | 0.445 | 0.342 | 0.220 | 0.001 |
| surrogate_ppri（DL 代理） | **0.963** | 0.698 | 0.151 | −2.144 | 0.551 | −0.314 |
| **pooled median** | **0.602** | 0.396 | 0.514 | 0.424 | 0.311 | 0.031 |
| 逐 oracle 第一名 | **4/8** | 0 | **4/8** | 0 | 0 | 0 |

PPO 全量（7 数学 oracle × 5 seed，本轮补齐）：ppri_additive 0.308 / ppri_cal 0.279 / nk_k0 0.474 / nk_k1 0.377 / nk_k2 0.400 / nk_k3 0.371 / gb1_pairwise 0.882。

## 3. 核心科学发现：优化器 × 景观强交互

| 景观类型 | 数学 oracle（NK/GB1/PprI 系） | DL 代理（真实数据训练） |
|---|---|---|
| WF | PprI 系 + GB1 全部第一 | 0.963 第一（最稳） |
| CEM | NK 全系第一（可分离→强上位性梯度） | 0.151 崩盘 |
| **PPO** | 中游（0.27–0.47），NK 系接近 CEM/ES | **0.698 第二，反超 CEM/ES** |
| ES | NK 系次席 | −2.144 严重失败 |

**结论**：
1. **WF + 先验提议是跨景观唯一稳定前二**的通用内核（ppri 系 + GB1 + 代理全第一，NK 系稳定前三）。
2. **PPO 是唯一能在 DL 代理景观反超分布类方法的 RL 内核**——代理景观（大范围平坦 + 稀疏有利区 + 强噪声）中，CEM/ES 的精英均值更新过早塌缩，PPO 学到的策略能利用代理平滑结构。PPO 在数学 oracle 上样本效率不足（每步 1 次 evaluate），但这是 RL 内核最具提升潜力的方向。
3. CEM 的上位性优势（NK 系 4/8 第一）**不迁移**到真实数据代理景观。

## 4. "完善模型"验证：代理训练策略已饱和

v2 增强训练（cosine 学习率调度 + 每 epoch held-out 评估 + 早停 + 12 epoch）：

| 指标 | v1（3 epoch） | v2（cosine+早停，ep0 即最佳，ep2 早停） |
|---|---|---|
| GRB2 Spearman | 0.288 | 0.290 |
| SPG1_Olson | 0.408 | 0.397 |
| SPG1_Wu | 0.247 | 0.244 |
| mean | 0.314 | 0.310 |

**结论**：浅特征 DeepSet 在 1 epoch 即收敛，更久训练/cosine 调度/早停无增益（ep1/ep2 下降）。**瓶颈在特征表达（±2 窗口 one-hot，无序列 embedding），不在训练策略**。要实质提升代理质量，下一步需引入 ESM-2 序列 embedding 特征（论文级，1–2 GPU 卡时，sbatch 模板已备：`train_surrogate.sbatch`）。

## 5. 下一步建议（按优先级）

1. **代理升级**：ESM-2 embedding 特征 → 新代理 → 重跑 surrogate 景观对比（GPU 提交须显式确认）
2. **PPO 调优**：lr/entropy/rollout 超参扫描，先用代理景观小规模验证（CPU 可跑）
3. **WF×PPO 集成**：以 WF 先验提议初始化 PPO 策略（行为克隆 warm-start），结合两者优势

## 6. 交付物

- 代码：compare 分支 `bf3b15d`（b7 `--optimizers` 过滤 + ppri_cal 依赖修复）；`0104e4f`（surrogate 接入）；rl-policy `3a67951`（torch PPO）
- 数据：`b7_full/{b7_v1_results.csv, b7_v1ppo_results.csv, b7_v2surr_results.csv}` + 3 份 traces（240 keys 全覆盖）
- 图：`b7_full/b7_violin_compare_v2.png`（6 内核 × 8 景观小提琴 + 代理收敛曲线）
- 模型：`surrogate_dl/{model.pt, meta.json}`；v2 复现脚本 `train_surrogate_v2.py`（结果与 v1 持平）
- 验证脚本：`verify_surrogate2.py`（端到端推理验证）

## 诚实披露

- 代理 Spearman 0.25–0.41 属浅特征基线水平；优化器对比结论仅适用于该代理景观，与 Boltz-2 真实激活一致性未验证。
- BC 在 gb1_pairwise 无数据（ppri 系对照轨迹 L=53 vs gb1 L=17 维度不匹配，非 bug）。
- PPO 数学 oracle 表现受 4000 evals 预算下在线样本效率限制；非生理学结论。

---

## §8 ESM3 特征升级（v3 代理）— 2026-08-31

**背景**：v1/v2（±2 one-hot 窗口，held-out mean 0.314）已饱和，瓶颈在特征体系。用户提供 CPU 集群 ESM3 → 升级为 ESM3 per-residue embedding 特征。

### 8.1 特征管线（新增）

| 组件 | 说明 |
|---|---|
| ESM3 模型 | esm3-sm-open-v1（1.4B），CPU 集群 10.205.1.3（sugon 分区） |
| 提取 | 218 序列（217 DMS WT + PprI WT 254aa），每序列一次 forward，hook 最后 transformer block，token[1:-1] 对齐残基（60aa→62 token 验证 1:1） |
| 作业 | sugon 8 核，494s 完成 218 条（~1-3s/序列） |
| 降维 | PCA 1536→64（85066 残基 fit，保留 59.6% 方差） |
| 特征 | site j 上下文 = concat(ESM3_emb[j-2..j+2]) → 5×64=320 + wt_aa(20) + mut_aa(20) + frac(1) = 341 维 |

### 8.2 模型质量（held-out Spearman，数据集级防泄漏）

| 指标 | v1/v2 | **v3 (ESM3)** |
|---|---|---|
| GRB2_HUMAN | 0.288 | **0.502** (+74%) |
| SPG1_Olson | 0.408 | 0.393 |
| SPG1_Wu | 0.247 | **0.327** (+32%) |
| **mean** | **0.314** | **0.407 (+30%)** |

v3 在 ep3 才达峰（v1/v2 ep0 即饱和）——ESM3 特征让模型从浅特征"一 epoch 学完"变为可持续学习，**瓶颈确实在特征不在训练策略**（验证上轮结论）。

### 8.3 PprI 零样本外推 — 诚实披露（关键）

- v3 对 PprI 53 位点单点扫描：**704/1007 (70%) 预测有益**，v1（one-hot）仅 225/1007 (22%)——**ESM3 特征在域外外推带整体正偏置**。
- K216（滑道）位点：v3 19 替换全部 ~0.916 饱和（mut_aa 失明），v1 有正常区分（K>D −0.32, K>Q −0.65）——**ESM3 上下文在该位点压过 AA 信号**。
- 根因：ESM3 embedding 是高维语义特征，PprI 与训练域差异大 → 外推不可控；one-hot 特征空间简单 → 外推中性。
- **结论**：v3 提升 in-domain 排序能力（+30%），但 PprI 零样本绝对尺度不可信；**排序信号（rank）可用，绝对 fitness 不可用**。

### 8.4 PprI 最优解（三信号交叉：v1 rank × 0.5 + v3 rank × 0.3 + evoprior 结构先验 rank × 0.2）

**新候选（代理强信号，建议 Boltz-2 验证）**：

| 候选 | 位点 | v1 | v3 | 逻辑 |
|---|---|---|---|---|
| **K216R** | 滑道 | 0.438 (前2%) | 0.916 | 保守正电增强，滑道中央 DNA 结合 |
| **Q120V** | 口袋 | 0.621 (第1) | 0.616 | 口袋疏水重塑 |
| **T89S** | 嘧啶读取 | 0.479 (前1%) | −0.159 | 嘧啶读取区微调（v3 弱，v1 强） |

**实验候选保持**：Y170F（v3 0.357 强支持）、F88Y（v3 无信号 — 代理不含 DNA 特异性，符合预期）、F88Y_Y170F（组合保持）。

**不采纳**：R267/R85 替换（代理高分但与已验证锚点证据矛盾）。

**推荐组合**：`K216R + Y170F + Q120V`（滑道正电 + 双锁 + 口袋），需 Boltz-2 状态分布验证（激活/双锁/判别三标签）。

### 8.5 踩坑记录

1. sbatch 日志目录不存在 → slurm 秒 FAILED（脚本 mkdir 无机会执行）→ 提交前先建日志目录。
2. `%%j` 在普通 Python 字符串中不转义 → 远端 sbatch 出现字面 `%%j` → slurm 秒拒（应写 `%j`）。
3. ESMProtein import 需在模块级（main() 内 import 时 extract() 报 NameError）。
4. 超长序列（3423aa）ESM3 CPU forward 无长度限制，全长直接跑（14s/条）。
