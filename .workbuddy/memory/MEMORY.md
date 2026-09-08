# PprI 项目状态摘要（2026-09-07 整理，后续对话以此为唯一基线）

## 当前目标
1. **主线**：PprI 特异性改造 → 湿实验验证 TrackF_F88W_M255A（唯一主推候选）
2. **副线**：EvoTrace 预测器生产化（v4+DL hybrid 互补评分，已定稿）

## 关键决策核心（已定，不推翻）
- **MD 战役判决**（PB 口径，ΔG_spec=PB(OFF)−PB(S1)，正值=靶标更稳）：**TrackF_F88W_M255A +16.08 ± 4.10（3.9σ，g=19.7 自相关校正，N_eff≈10；naive 17σ 已弃用）唯一真特异**（GB +27.67 双法方向一致）；s13_c1 −5.82 ❌ / w4 −8.18 ❌（Boltz sep 0.53 被 MD 证伪）；RD_POS −12.28 强结合无选择性=定标成功。失败模式统一=OFF poly-T 被缠绕（vdw 驱动）。
- **GB vs PB 差 ~100 kcal 已确诊非故障**：① PB 段 EEL=真实库仑÷indi(=4.0)（全体系比值精确 4.000）；② GB(igb=5) 对蛋白-DNA 极性罚分系统性偏大（EGB diff +257 vs EPB +67）。**论文只报 PB 相对值**；GB+idecomp 仅作残基分解机制证据。
- **WT 基线对比**（S1 −51.54±9.41，按"WT 无特异性"默认前提兼任 OFF 代理）：TrackF 保留 50% 靶标亲和、OFF 臂损失 42.1 kcal → 教科书式特异化；w4 WT 型平坦（+11.1/+2.9）；s13 双弱偏 OFF（最差）；RD_POS 双强更偏 OFF。
- **nom2 结论（n=100）**：M255I 组合全部出局；F88R 单点证伪；湿实验提名以 TrackF F88W_M255A（Δ+0.24 z=3.93）为首。
- **EvoTrace 模型线**：v7 DL 赢 act 0.770/dual 0.838/dcat 0.748，输 sep（−0.124 vs v4 +0.330）→ **生产方案=v4+DL hybrid**（min(rank(dl_act),rank(v4_sep))，rho +0.630），DL 管结合强度、v4 管特异性。

## 进度约束（截至 9/7）
- **MD 全部收官，数据审计通过**（audit_md_results_20260907.md）：8 条 prod.nc(~8.5G)+topol+PBSA 全在本地 `A:\claudework\设计蛋白\PprI_ssDNA_design\md_specificity\cluster_mirror\`；WT 分析级在 `results\WT__S1\`；257 张图在 `mdfigs\`。**唯一缺口=WT prod.nc 仅在 CHPC Lustre**（建议快照备份）。
- 4090D 实例 21114 任务完毕可关机。
- 设计蛋白资料库已整体迁入 `设计蛋白/`（含 MD 数据/FoldX/AF3/论文，自身独立 git，已 gitignore；旧路径 A:\Data\设计蛋白 只剩空壳留有 MOVED 说明）；项目目录 9/7 已清理：无关脚本/scratch/旧方案移入 `_archive/2026-09-07_cleanup/`（未删除）；旧日志移入 `.workbuddy/memory/archive/`。

## 下一步行动
1. **湿实验执行 TrackF_F88W_M255A**（唯一主推）
2. 可选加固：TrackF ΔG_spec 分块收敛分析（用集群 keep_files 逐帧数据）；补测 WT_OFF 轨迹；CHPC WT prod.nc 快照
3. **EvoTrace 主线收敛路线 B（9/8 并行会话 b11/b12 判决 + 9/9 路线图闭环）**：b10(均匀混合)→b11(岛屿)→b12(耦合入适应度) 三代阴性证据链闭合——"DCA 二阶耦合对共识距离二分回放外推无增益"为可发表阴性结论；L3 不再是 M2 blocker，主线=v4+DL hybrid；路线图 R4 已关账（32e9544）

## 资源/约定（长期有效）
- Boltz-2 相关任务（含 CHPC GPU 提交）**永久免确认**；其余 GPU/大批量任务须显式确认
- CHPC 登录 10.202.94.52:20009（家目录 /home/u22607007）；4090D 实例 10.202.94.52:21114；CPU 集群 10.205.1.3:**10022**（勿连 22），密钥 `A:/edge/文献/10.205.1.3_0826123315_rsa.txt`，家目录 /public/home/u22607007
- CHPC SFTP put 报 ENOENT → 走 base64 通道；Lustre 作业产物在 /users/u22607007/...（探针先 `df -h` 查挂载）
- Boltz 判读铁律：HEXXH(seq71-75) 接触 + K67-G17/R232-T23 ≤5Å + S1−OFF 判别；seq=PDB−21
- scan_n100_full.py / scan_nom2_full.py 保留根目录可复用；mdtraj shrake_rupley 单帧损坏须跑 repair_sasa.py；MD 图规范：WT 黑基线、RMSD 2/3/5Å 判读带、PBSA ±SEM、氢键 30% 线
