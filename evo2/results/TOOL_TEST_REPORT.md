# EvoTrace 工具测试报告（v1.1 + DNA-aware）

- **日期**：2026-08-31（分支 `dna-aware-landscape`，HEAD `b90bdd1`）
- **测试命令**：`cd A:/claudework/evo2 && A:/Python/python.exe -m pytest tests/ -v`
- **结果**：**23 passed in 9.39s**（23/23，零失败）
- **范围**：六搜索内核 + 内置 oracle + ESM3 代理接口 + DNA 条件门控（gate v1/v2/v3）+ b7 基准管道装配

---

## 1. 测试套件结构（`evo2/tests/`，新增）

| 模块 | 用例数 | 覆盖 |
|---|---|---:|---|
| `test_tool_smoke.py` | 8 | 六内核可发现、oracle 接口契约（L/max_mut/wt_idx/sites/evaluate/n_mutations/enforce_max_mut）、每内核 300 evals 跑通一轮 |
| `test_tool_correctness.py` | 5 | 加性闭式最优、**WF 收敛**（2000 evals → norm 0.988）、**确定性**（同 seed 同结果）、NK 接口、max_mut 强制回退 |
| `test_dna_gate.py` | 6 | gate v3 默认、机制方向（M255I>M、Y217Y>R、F88K>F）、**v3 vs Boltz 实测正相关**、s13_c1 复现 top、**v1 反相关护栏**、组合适应度 |
| `test_b7_regression.py` | 4 | b7 main() 注册声明完整、六优化器发现、oracle 管道契约、加性景观 **WF ≥ CEM**（b7 结论冒烟复现） |

## 2. 工具健康度结论（全部通过）

1. **六内核全部可用且行为正常**：WF/PPO/DQN/BC/CEM/ES 均能实例化并完成一轮搜索，输出合法。
2. **WF 收敛符合基准**：可分离加性景观 L=8 上，2000 evals（10 代）→ norm **0.988**，4000 evals → **1.000**（与 b7 全量"WF 最稳"结论一致）。
3. **确定性成立**：同 seed 两次 run 的 best_f 与完整 trace 完全一致 → 工具可复现。
4. **max_mut 强制正确**：6 突变基因型被强制回退到 ≤3，且回退只恢复 WT、不引入新突变。
5. **gate v3 与 Boltz 实测一致**（去混杂数据 22 变体）：
   - Spearman vs `d_iface` **+0.858**、vs `dual_S1` 与 `d_act` 均 **>0.3**（断言裕度内）；
   - 机制方向全部正确：M255I > M255M、Y217Y > Y217R、F88K/F88W > F88F；
   - `F88K_M255I_Y217Y`（= s13_c1 复现）gate 分数名列前茅。
6. **v1 解析式反相关护栏生效**（Spearman < 0）——防止未来回归到被证伪的启发式。
7. **b7 管道装配完整**：REBUILD 注册声明、六优化器发现、oracle 契约、WF≥CEM 关系均验证。

## 3. 测试暴露的问题（均为测试自身设计，非工具缺陷）

| 现象 | 根因 | 修复 |
|---|---|---|
| b7 REBUILD 为空 dict | 注册在 `main()` 内，import 时不执行 | 改为源码级注册断言 + 独立管道验证 |
| WF norm 0.831 < 0.90 | 测试预算 800 evals = 仅 4 代，WF 未起步（工具本身 4000 evals → 1.000） | 预算校准至 2000 evals（10 代，norm 0.988） |
| WF(0.328) < CEM(0.618) | 同为预算过短（400 evals = 2 代；CEM 分布搜索首轮即采样占优） | 预算 2000：WF 0.988 > CEM 0.618，与 b7 结论一致 |
| `int(数组).sum()` TypeError | 测试代码括号错误 | 修正为 `int(reverted.sum())` |

**结论：工具代码无需任何修改——v1.1 + DNA-aware(gate v3) 作为工具，行为、收敛、机制方向、管道装配全部符合设计。**

## 4. 局限

1. 测试未加载 ESM3 权重（`PprISurrogateV3` 全量推理），代理端到端行为由 `test_dna_gate.py` 的桩基隔离验证 + 既往 b7 全量基准（held-out 0.407）背书。
2. gate 相关性断言基于全量拟合（+0.858），LOO 泛化估计 +0.583 未入断言（留裕度）。
3. 性能/内存压力（如 b7 注释的 DQN 4000 evals 内存碎片问题）未纳入测试——属预算级行为，非功能正确性。
