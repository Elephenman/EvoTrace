# pform — protein final-state predictor

EvoTrace 集群工具套件的端到端"蛋白最终形态预测"CLI（蛋白无关，指标可插拔）。

## 定位

把"输入蛋白设计 → 预测最终形态（结构 + 功能指标 + 校准裁决）"变成一条命令：
上传 yaml → CHPC 4090 提交 Boltz-2 → 轮询 → 远程指标计算 → 回拉 CSV/JSON → 最终形态裁决。

## 快速开始

```bash
export CHPC_PASS='...'          # 集群密码（不落盘）

# 1) 基础预测（20 模型/候选）
python predict.py predict <yamls_dir> --job my_run --wait

# 2) 结合态模式（模板引导，推荐）：给 holo 晶体模板，测"结合后形态"
python predict.py predict <yamls_dir> --template 8SLN.cif --template-chain A \
    --job binding_run --wait

# 3) 多 seed 稳健性（3 seed × 20 = 60 模型/候选）
python predict.py predict <yamls_dir> --seeds 1 2 3 --template 8SLN.cif \
    --job multi --wait --out-csv results/models.csv

# 4) 校准：对照已知实验真值
python predict.py calibrate results/models_s1.csv --truth truth.json
```

## 子命令

| 命令 | 功能 |
|---|---|
| `predict` | 全流程：上传 → sbatch → 轮询 → 指标 → 回拉 → 汇总 |
| `status [--job X]` | 查看队列/进度（cif 数） |
| `metrics --job X [--seeds 1 2 3]` | 只重算指标（不重跑 GPU） |
| `calibrate <csv> --truth <json>` | 对照真值输出校准（bias / correction_factor） |

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--samples` | 20 | 每候选 diffusion 采样数 |
| `--seeds` | 1 | seed 列表（如 `1 2 3`，每个独立采样） |
| `--template <cif>` | — | 结合态模板（holo 晶体），自动注入 yaml `templates` 段 |
| `--template-chain` | A | 模板链映射到输入蛋白的链 |
| `--wait` | off | 提交后轮询到完成并自动算指标 |
| `--timeout-min` | 480 | 等待超时 |
| `--out-csv` | results/ | 回拉 CSV 路径 |

## 指标（可插拔）

默认 `metrics_dual_lock.py`：锚点对距离（P1/P2）+ 催化域距离 + 金属距离 + 二值裁决（dual）。
`DEFAULT_CFG` 可配链/锚点/阈值 → 任意蛋白-核酸-配体复合物。
自定义指标：新写一个 `build_remote_script()`（纯 stdlib，集群登录节点运行，只回拉 CSV）。

## 校准输出（诚实边界）

```json
// truth.json —— 已知实验真值
{"WT": {"expected_dual": 1.0, "note": "8SLN crystal dual-locked"}}
```
输出：`CAL exp=1.00 bias=-0.70 factor=3.33` —— 预测要乘 3.33 才是真值期望。
无真值候选标注 `uncalibrated`。**工具输出相对排序 + 校准置信度，不做绝对判决。**

## 环境与已知坑

- 集群：`10.202.94.52:20009` 用户 `u22607007`，密码 `CHPC_PASS`
- Boltz-2 必需配置（已内嵌 sbatch）：`NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib`、
  `HF_ENDPOINT=https://hf-mirror.com`、`--no_trifast`、`--use_msa_server`（ColabFold 公共 server）
- **Boltz 2.0.3 的 yaml `constraints` 节不可用**（contact 解析 bug + pocket 不激活）→ 用 `templates` 代替
- sbatch 必须 LF 行尾（自动归一化）；上传走 SFTP+base64 双通道
- 观测结论：无模板时 Boltz 测"结合前采样"（WT 0/20）；8SLN 模板引导后测"结合后形态"（WT 6/20, 0.30）→ **设计相对排序可靠，绝对判决需校准**

## 文件

```
pform/
├── predict.py            # CLI 入口（predict/status/metrics/calibrate）
├── chpc.py               # paramiko 客户端（SFTP+base64、LF 归一化）
└── metrics_dual_lock.py  # 默认可插拔指标
```
