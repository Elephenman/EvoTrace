# 项目当前状态（PROJECT_STATE，2026-09-07）

> 本文件是项目唯一基线摘要。历史细节在 `_archive/2026-09-07_cleanup/` 与 `.workbuddy/memory/archive/`，均未删除。

## 当前目标
**PprI 特异性改造**：MD 验证已完成并判决，已生成湿实验执行包但**尚未执行湿实验**，主推唯一计算候选 **TrackF_r1::K88W/M255A**。
旧别名 `TrackF_F88W_M255A` 与 `TrackF_r1_F88W_M255A` 仅用于 legacy 外键迁移，不再作为当前候选身份。
副线 EvoTrace：模型选型定稿（v4+DL hybrid），无紧急项。

## 核心结论（MD 战役，PB 口径 ΔG_spec = PB(OFF)−PB(S1)）
| 体系 | ΔG_spec | 判定 |
|---|---|---|
| **TrackF_r1::K88W/M255A** | **+16.08 ± 4.10（3.9σ，自相关校正 N_eff≈10；naive ±0.96/17σ 已弃用，9/7 收敛分析修正）** | ✅ 当前唯一计算优先候选（GB 仅作机制解释） |
| RD_POS（对照） | −12.28 | 强结合无选择性，定标成功 |
| s13_c1 | −5.82 | ❌ 出局 |
| w4 | −8.18 | ❌ 出局（Boltz sep 0.53 被证伪） |

WT 基线（S1 −51.54，默认无特异性兼任 OFF 代理）：TrackF 保留 50% 靶标亲和、OFF 臂损失 42.1 kcal —— 特异化机制 = 不成比例削弱非靶臂。
GB/PB 差 ~100 kcal 已确诊为模型系统偏差（indi=4 记账 + GB 核酸失效），非数据问题；论文只报 PB 相对值。

## 数据地图
| 内容 | 位置 |
|---|---|
| 8 条原始轨迹+拓扑+PBSA（~8.5G，审计通过） | `A:\claudework\设计蛋白\PprI_ssDNA_design\md_specificity\cluster_mirror\` |
| WT 分析级数据（无原始轨迹，仅 CHPC 有） | `设计蛋白\PprI_ssDNA_design\md_specificity\results\WT__S1\` |
| 9 个 npz + 257 张图 | `设计蛋白\PprI_ssDNA_design\md_specificity\mdfigs\` |
| MD 审计报告 / GB-PB 诊断 | `设计蛋白\PprI_ssDNA_design\md_specificity\audit_md_results_20260907.md` |
| Boltz nom2 报告 | `evo2/results/boltz_nom2/REPORT.md` |
| EvoTrace 路线图与提案 | `evotrace_restore/` |
| 归档（旧脚本/scratch/技术方案 v1） | `_archive/2026-09-07_cleanup/` |
| 设计蛋白资料库（MD/FoldX/AF3/论文，2026-09-07 自 A:\Data 整体迁入） | `设计蛋白/`（自身含独立 git 仓库，已加 .gitignore） |

## 下一步行动
1. 待用户确认后执行 TrackF_r1::K88W/M255A 湿实验（执行包已就绪但尚未执行；旧文件名保留作迁移外键）
2. ✅ 已完成（9/7 收敛分析；9/9 凌晨-上午）：TrackF ΔG_spec 收敛分析（±4.10/3.9σ, 图 c18）→ **MD 结构检视**（8 体系 start/150ns 端点 PDB + PyMOL 12 图；s13 S1 臂 DNA 完全解离/TrackF 锚点合拢，报告 `md_specificity/structure_views/`）→ **MD 数据归档**（`设计蛋白/MD数据归档_20260909/`，7.57GB/471 文件零差异，含归档清单）；WT prod.nc 快照行动关闭（CHPC 副本已被清场）；全局 MemoryGate hook 中文路径编码 bug 已修复（三层：UTF8 解码/乱码自愈/优雅降级）
3. 待定（需确认）：补测 WT_OFF 150ns MD（消除 WT 代理基线假设）——4090D 实例 21114 SSH 无响应需先重启实例，或等 CHPC gpu 分区空闲；CD 由用户决定是否投入
4. EvoTrace：按 evotrace_restore/ 路线图择机推进

## 长期约定
- 所有 GPU/集群任务（含 Boltz-2）均须显式确认；当前版本只保存 payload，不提交作业
- 集群：CHPC 10.202.94.52:20009（/home/u22607007）｜4090D 实例 :21114｜CPU 集群 10.205.1.3:10022（/public/home/u22607007）
- 完整约定见 `.workbuddy/memory/MEMORY.md`（唯一记忆基线）
