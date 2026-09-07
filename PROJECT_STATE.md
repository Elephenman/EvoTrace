# 项目当前状态（PROJECT_STATE，2026-09-07）

> 本文件是项目唯一基线摘要。历史细节在 `_archive/2026-09-07_cleanup/` 与 `.workbuddy/memory/archive/`，均未删除。

## 当前目标
**PprI 特异性改造**：MD 验证已完成并判决，转入湿实验执行，主推唯一候选 **TrackF_F88W_M255A**。
副线 EvoTrace：模型选型定稿（v4+DL hybrid），无紧急项。

## 核心结论（MD 战役，PB 口径 ΔG_spec = PB(OFF)−PB(S1)）
| 体系 | ΔG_spec | 判定 |
|---|---|---|
| **TrackF_F88W_M255A** | **+16.08（~17σ）** | ✅ 唯一真特异（GB +27.67 双法一致） |
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
1. 湿实验执行 TrackF_F88W_M255A
2. 可选加固：TrackF ΔG_spec 分块收敛分析（集群逐帧数据）；补测 WT_OFF；CHPC WT prod.nc 快照
3. EvoTrace：按 evotrace_restore/ 路线图择机推进

## 长期约定
- Boltz-2 任务免确认；其余 GPU 大批量须显式确认
- 集群：CHPC 10.202.94.52:20009（/home/u22607007）｜4090D 实例 :21114｜CPU 集群 10.205.1.3:10022（/public/home/u22607007）
- 完整约定见 `.workbuddy/memory/MEMORY.md`（唯一记忆基线）
