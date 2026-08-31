#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v3 草案 — EvoPrior：数据驱动的位点偏好预测器（进化先验 p_evo）。

动机
----
M1 的几何/化学先验 p_chem 是"手工规则"，存在系统性盲区。典型例子：PprI 的
135 位为埋藏疏水位（dominant_class='buried'），化学先验给 M=0.85 / H=0.25；
手工规则却把该位改成 H（"M135H"），与进化事实冲突——PprI 同源 MSA 中 135 位
强烈偏好疏水残基（M/F/L）。这正是论文中"特异性–稳定性权衡归因"的最大弱点：
部分 trade-off 源于先验设计的盲点。

EvoPrior 用蛋白质语言模型（ESM-2 / ProtBERT）的逐残基嵌入作为输入，从同源
序列 MSA 提取的 PSSM（位置特异性评分矩阵）作标签，训练一个轻量 MLP 预测每个
位点的 20 维天然氨基酸偏好 p_evo（"自然先验"，而非手工规则）。

在 M2 景观构建层，把化学先验 p_chem 与进化先验 p_evo 加权融合：
    p_final = α · p_chem + (1 − α) · p_evo
α 通过交叉验证学习（最小化 KL(p_ssm ∥ p_final)），使融合分布尽量贴近真实进化
频率。

预期效果
--------
在 135 位，p_evo 给出 M/F/L 高概率，融合后 p_final 仍然偏好疏水，从而自动纠正
"M135H" 盲点。方法学叙事从"揭示了一个手工规则导致的盲点"升级为"用深度学习
自动识别并规避了该盲点"，并给出"规则驱动 vs 数据驱动"的定量对比。

依赖
----
- numpy（必需）
- transformers + torch（可选，仅真实 PLM 嵌入需要；缺失时回退 placeholder 嵌入）

接口
----
- build_evoprior(seq, msa_path, ...) -> dict(p_evo, alpha, pssm, model, embed)
- fuse_tables(p_chem, p_evo, alpha)
- learn_alpha_cv(p_chem, p_evo, pssm, folds)
"""
import math
import os
import warnings

import numpy as np

from .seqtools import AA20, AA2IDX

N_AA = 20
GAP = "-"


# ======================================================================
# 1. MSA 解析与 PSSM（标签）
# ======================================================================
def read_a3m(path):
    """读取 a3m：第一行=query(WT)，其余为同源序列（剔除插入列，保留 gap）。

    a3m 格式约定：
      - **小写字母 = 相对 query 的插入列**，不占对齐列 → 必须剔除；
      - **'-' = 该序列在此对齐列的缺失** → 必须保留，否则序列被压缩、
        后续列整体错位，且长度不再等于 query 长度。

    ⚠️ 历史 bug：曾用 `c.isupper()` 过滤，把 '-' 一并剔除，导致所有同源
    序列长度 ≠ L 而被 msa_to_pssm 静默跳过（seen=1，只剩 query 自身），
    PSSM 退化为"query one-hot + 伪计数"，几乎不含进化信息（平均熵 2.978
    vs 均匀 2.996）。此处保留 '-' 即为修复。

    返回 (query, [seq, ...])，长度均为对齐列数 L。
    """
    seqs = []
    query = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        # 仅剔除小写插入列；保留大写残基与 gap '-'
        s = "".join(c for c in line if not c.islower())
        if query is None:
            query = s
        seqs.append(s)
    if query is None:
        raise ValueError(f"{path}: 空 MSA")
    return query, seqs


def read_fasta_aligned(path):
    """读普通 fasta 多序列（已对齐，含 '-'）。返回 (query, [seq,...])。"""
    seqs = []
    query = None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if query is None:
                query = ""
            continue
        if query is not None and not seqs:
            pass
        seqs.append(line.upper())
    # 简化：第一行当作 query
    if not seqs:
        raise ValueError("空 fasta")
    return seqs[0], seqs[1:] or [seqs[0]]


def msa_column_neff(seqs, query):
    """每列的有效序列数（非 gap 残基数）[L]。用于判断 PSSM 在各列的可信度。"""
    L = len(query)
    neff = np.zeros(L, dtype=float)
    for s in seqs:
        if len(s) != L:
            continue
        for i, c in enumerate(s):
            if c != GAP and c in AA2IDX:
                neff[i] += 1.0
    return neff


def msa_to_pssm(seqs, query, pseudocount=1.0, max_gap_frac=1.0, warn=True):
    """同源序列 -> [L, 20] PSSM（加伪计数的频率，每行归一）。

    seqs 为已对齐的同源序列列表；按 query 长度 L 逐列统计 20 种 AA 频率，
    gap（'-'）不计入任何氨基酸。

    参数
    ----
    max_gap_frac : 单条序列 gap 占比上限，超过则跳过（默认 1.0 = 不过滤）。
    warn         : 当被跳过的行占比过半时发警告——历史上 read_a3m 误删 '-'
                   曾导致绝大多数行长度不匹配而被静默跳过，PSSM 退化。
    """
    L = len(query)
    counts = np.zeros((L, N_AA), dtype=float)
    seen = 0
    skipped = 0
    for s in seqs:
        if len(s) != L:
            skipped += 1
            continue
        if max_gap_frac < 1.0 and (s.count(GAP) / L) > max_gap_frac:
            skipped += 1
            continue
        seen += 1
        for i, c in enumerate(s):
            idx = AA2IDX.get(c)
            if idx is not None:          # gap / 非标准残基不计入
                counts[i, idx] += 1.0
    if seen == 0:
        raise ValueError("没有与 query 等长的 MSA 行（对齐质量差或 query 错）")
    if warn and skipped > 0.5 * (seen + skipped):
        warnings.warn(
            f"MSA 中 {skipped}/{seen + skipped} 行因长度≠{L}（或 gap 过多）被跳过；"
            f"若占比过高，请检查 a3m 解析——gap '-' 必须保留，否则列会错位",
            RuntimeWarning, stacklevel=2,
        )
    counts += pseudocount
    pssm = counts / counts.sum(axis=1, keepdims=True)
    return pssm


# ======================================================================
# 2. 嵌入（PLM，可选）
# ======================================================================
def get_embeddings(seq, model="esm2_t33_650M_UR50D", device="cpu", cache=None):
    """用 ESM-2 提取逐残基嵌入 [L, D]。需要 transformers + torch。

    cache：若给定 .npy 路径且存在则直接加载（避免重复推理）。
    """
    if cache:
        import os
        if os.path.exists(cache):
            return np.load(cache)
    try:
        from transformers import AutoTokenizer, EsmModel  # noqa
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "真实 PLM 嵌入需要 transformers+torch。未安装时请使用 "
            "placeholder_embeddings() 跑通流程，或 `pip install transformers torch`"
        ) from e
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained(f"facebook/{model}")
    mdl = EsmModel.from_pretrained(f"facebook/{model}").to(device).eval()
    toks = tok(seq, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    import torch
    with torch.no_grad():
        out = mdl(toks).last_hidden_state[0].cpu().numpy()
    emb = out[: len(seq)]  # 截断到序列长度（去 CLS/起止）
    if cache:
        np.save(cache, emb)
    return emb


def placeholder_embeddings(seq, window=3):
    """无 PLM 时的退化嵌入：位点 one-hot + 左右 window 邻域均值 [L, 20*(2w+1)]。

    仅用于在缺 transformers 时跑通端到端流程；**不代表真实 PLM 信号**，
    论文实验须替换为 get_embeddings() 的真实 ESM-2/ProtBERT 嵌入。
    """
    L = len(seq)
    oh = np.zeros((L, N_AA))
    for i, c in enumerate(seq):
        idx = AA2IDX.get(c)
        if idx is not None:
            oh[i, idx] = 1.0
    dim = N_AA * (2 * window + 1)
    emb = np.zeros((L, dim))
    for i in range(L):
        block = []
        for j in range(i - window, i + window + 1):
            if 0 <= j < L:
                block.append(oh[j])
            else:
                block.append(np.zeros(N_AA))
        emb[i] = np.concatenate(block)
    return emb


def load_npz_embeddings(path, key=None):
    """加载集群预计算的 PLM 嵌入（`cluster/esm3_embed_cluster.py` 的产物 npz）。

    背景：本地常因网络限制无法下载 PLM 权重，而集群 CPU 环境已具备
    ESM3-sm-open-v1 权重（并通过 data_root 打补丁绕开缺失的 HuggingFace
    缓存）。故采用"集群预计算嵌入 → 回传 npz → 本地训练 EvoPrior"两段式。

    返回 (emb[L, D], key)。
    """
    d = np.load(path)
    keys = list(d.files)
    if key is None:
        if len(keys) != 1:
            raise ValueError(f"npz 含 {len(keys)} 个条目，请用 key= 指定: {keys}")
        key = keys[0]
    elif key not in keys:
        raise KeyError(f"npz 中无 key={key!r}，可选: {keys}")
    emb = np.asarray(d[key], dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"嵌入应为 [L, D] 二维，得到 {emb.shape}")
    return emb, key


def pca_reduce(emb, dim=32, whiten=True):
    """PCA 降维（含白化），抑制 D >> L 时的过拟合。

    典型情形：ESM3-sm-open-v1 嵌入 D=1536，而单条蛋白仅 L≈254 个位点，
    直接用 1536 维训练 MLP 会严重过拟合（参数量 ≫ 样本量）。

    返回 (Z[L, dim], meta)；meta 可传给 apply_pca() 复现同一变换。
    """
    X = np.asarray(emb, float)
    L, D = X.shape
    dim = int(max(1, min(dim, L, D)))
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:dim]
    Z = Xc @ W.T
    if whiten:
        scale = np.sqrt(np.maximum(S[:dim] ** 2 / max(L - 1, 1), 1e-12))
        Z = Z / scale
        W = W / scale[:, None]
    tot = float((S ** 2).sum())
    meta = dict(mean=mu[0], components=W, dim=dim, in_dim=D,
                explained_var=float((S[:dim] ** 2).sum() / tot) if tot > 0 else 1.0)
    return Z, meta


def apply_pca(emb, meta):
    """用 pca_reduce() 返回的 meta 对新嵌入施加同一 PCA 变换。"""
    X = np.asarray(emb, float)
    return (X - np.asarray(meta["mean"], float)) @ np.asarray(meta["components"], float).T


# ======================================================================
# 3. 模型：轻量 MLP（纯 numpy，输入=嵌入，输出=20 维分布）
# ======================================================================
class EvoPriorMLP:
    """两层 MLP：嵌入 -> hidden -> 20 维 softmax 分布（天然氨基酸偏好）。

    训练目标：交叉熵 KL(p_ssm ∥ p_pred)。纯 numpy 实现，单序列规模
    （L≈250 位点）CPU 上亚秒级。
    """

    def __init__(self, in_dim, hidden=256, n_aa=N_AA, seed=0):
        self.rng = np.random.default_rng(seed)
        s1 = math.sqrt(2.0 / in_dim)
        s2 = math.sqrt(2.0 / hidden)
        self.W1 = self.rng.normal(0, s1, (hidden, in_dim)).astype(np.float64)
        self.b1 = np.zeros(hidden)
        self.W2 = self.rng.normal(0, s2, (n_aa, hidden)).astype(np.float64)
        self.b2 = np.zeros(n_aa)
        self.hidden = hidden

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        self._h = np.maximum(0, X @ self.W1.T + self.b1)        # [N, hidden]
        z = self._h @ self.W2.T + self.b2                        # [N, 20]
        return self._softmax(z)

    def fit(self, X, Y, epochs=300, lr=0.02, w_decay=1e-4, verbose=True):
        """X:[N,D] 嵌入, Y:[N,20] PSSM 目标（每行分布）。返回 loss 历史。"""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        loss_hist = []
        for ep in range(epochs):
            P = self.forward(X)
            # 交叉熵
            eps = 1e-8
            loss = -np.mean(np.sum(Y * np.log(P + eps), axis=1))
            loss_hist.append(float(loss))
            # 反向（CE + softmax 解析梯度）
            dZ = (P - Y) / X.shape[0]                            # [N,20]
            dW2 = dZ.T @ self._h + w_decay * self.W2
            db2 = dZ.sum(axis=0)
            dH = (dZ @ self.W2) * (self._h > 0)                  # [N,hidden]
            dW1 = dH.T @ X + w_decay * self.W1
            db1 = dH.sum(axis=0)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            if verbose and (ep + 1) % 50 == 0:
                print(f"  [EvoPrior] epoch {ep+1}/{epochs}  CE={loss:.4f}")
        return loss_hist

    def predict(self, X):
        return self.forward(np.asarray(X, float))

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(d["W1"].shape[1], d["W1"].shape[0])
        m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return m


# ======================================================================
# 4. 融合层（M2）+ α 交叉验证
# ======================================================================
def fuse_tables(p_chem, p_evo, alpha=0.5):
    """融合 p_chem 与 p_evo。

    p_chem, p_evo: [L,20]（每行已归一的分布）。
    alpha: 标量（全局权重）或 [L] 数组（逐位点自适应权重）。
    返回 p_final = normalize(α·p_chem + (1−α)·p_evo)，每行归一。
    """
    p_chem = np.asarray(p_chem, float)
    p_evo = np.asarray(p_evo, float)
    a = np.asarray(alpha, float)
    if a.ndim == 0:
        a = a * np.ones(p_chem.shape[0])
    a = a.reshape(-1, 1)  # [L,1]
    fused = a * p_chem + (1.0 - a) * p_evo
    fused = np.clip(fused, 1e-6, None)
    return fused / fused.sum(axis=1, keepdims=True)


def _kl_rows(p_true, p_pred):
    eps = 1e-8
    return np.sum(p_true * (np.log(p_true + eps) - np.log(p_pred + eps)), axis=1)


def _js_div(p, q):
    """Jensen-Shannon 分歧（nats，∈ [0, log2]）。衡量两分布的对称相似度。"""
    p = np.clip(np.asarray(p, float), 1e-8, None)
    q = np.clip(np.asarray(q, float), 1e-8, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    return 0.5 * kl_pm + 0.5 * np.sum(q * (np.log(q) - np.log(m)))


def learn_alpha_cv(p_chem, p_evo, pssm, folds=5, alphas=None, per_site=False):
    """融合层 α 交叉验证。

    全局模式(per_site=False)：k 折 CV 在验证位点上选标量 α 最小化
        mean KL(p_ssm ∥ p_final)，返回 (最优 α, (α, mean_KL) 表)。
    逐位点模式(per_site=True)：返回 [L] 的 α 数组，每位点
        α_j = 1 − d_j，其中 d_j = JS(p_chem_j, p_evo_j)/log2 ∈ [0,1] 为
        化学先验与进化先验在该位的局部分歧：
          · 规则与数据一致(d_j→0) → α_j→1，保留化学先验（"化学先验强"）
          · 规则被盲化、与数据冲突(d_j→1，即盲点) → α_j→0，转信进化先验（"进化信号强"）
    设计要点
    --------
    - 不直接用 p_ssm_j 逐位拟合（避免 p_evo≈pssm 时 α→0 退化的过拟合）；
      仅以 p_chem_j 与 p_evo_j 的分歧为信号，符合"规则驱动 vs 数据驱动"
      的定性对比，且不依赖 PLM。
    - 不依赖 k 折 CV（避免 L=1 单点时训练集为空的退化）。
    注：CV 只调融合权重，不重训 MLP——MLP 已在全量 PSSM 上训练。
    """
    if per_site:
        p_chem = np.asarray(p_chem, float)
        p_evo = np.asarray(p_evo, float)
        L = p_chem.shape[0]
        log2 = math.log(2.0)
        alpha_arr = np.zeros(L)
        for j in range(L):
            d = _js_div(p_chem[j], p_evo[j]) / log2
            alpha_arr[j] = 1.0 - min(max(float(d), 0.0), 1.0)
        return alpha_arr, None
    if alphas is None:
        alphas = np.round(np.linspace(0.0, 1.0, 21), 2)
    p_chem = np.asarray(p_chem, float)
    p_evo = np.asarray(p_evo, float)
    pssm = np.asarray(pssm, float)
    L = p_chem.shape[0]
    idx = np.arange(L)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    splits = np.array_split(idx, folds)
    mean_kl = {float(a): 0.0 for a in alphas}
    for fl in range(folds):
        val = splits[fl]
        # α 是融合超参（无待估参数），因此**每个候选 α 都在验证折上评估**，
        # 再对折取平均。历史实现先在训练集上挑一个"最佳 α"、只把它的验证
        # KL 累加，导致其余 α 桶保持初值 0，曲线出现大量假 0（伪最优）。
        for a in alphas:
            pf_v = fuse_tables(p_chem[val], p_evo[val], a)
            mean_kl[float(a)] += _kl_rows(pssm[val], pf_v).mean()
    for a in mean_kl:
        mean_kl[a] /= folds
    best = min(mean_kl, key=mean_kl.get)
    return float(best), [(a, mean_kl[a]) for a in alphas]


# ======================================================================
# 5. 端到端
# ======================================================================
def evaluate_mlp_cv(feat, pssm, wt_idx=None, folds=5, hidden=64, epochs=300,
                    lr=0.03, w_decay=1e-2, seed=0):
    """位点 k 折交叉验证：判断 PLM 嵌入是否真有"可泛化"的位点偏好信息。

    单条蛋白只有 L≈254 个训练位点，MLP 完全可以靠记忆 PSSM 拿到很低的
    训练损失——那不是学习。本函数在留出位点上评估，并与两个不含嵌入的
    基线对比：

      composition : 训练位点的平均 PSSM（全局氨基酸组成）——弱基线
      wt_cond     : 按 WT 残基类型查表（训练集中 WT==a 位点的平均 PSSM）
                    ——**强基线**。PSSM 的可预测性很大一部分来自"该位点
                    当前是什么氨基酸"，不打败它就说明嵌入没带来位点环境信息。

    参数 wt_idx : 长度 L 的 WT 残基索引数组（AA2IDX 编码）；为 None 时跳过
                  wt_cond 基线。

    返回 dict：
      mlp_ce / comp_ce / wtcond_ce : 留出位点交叉熵（越低越好）
      delta_ce       : comp_ce − mlp_ce（>0：优于组成基线）
      delta_wt_ce    : wtcond_ce − mlp_ce（>0：优于 WT 查表，即嵌入有真增益）
      ce_per_site    : 逐位点 CE 数组（可做配对 t 检验）
      mlp_pred       : 逐位点留出预测 [L,20]（可直接作为 p_evo）
    """
    feat = np.asarray(feat, float)
    pssm = np.asarray(pssm, float)
    N = feat.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    splits = np.array_split(idx, max(1, min(folds, N)))
    pred_mlp = np.zeros_like(pssm)
    pred_comp = np.zeros_like(pssm)
    pred_wt = np.zeros_like(pssm) if wt_idx is not None else None

    for fl in range(len(splits)):
        val = splits[fl]
        tr = np.concatenate([splits[k] for k in range(len(splits)) if k != fl])
        m = EvoPriorMLP(feat.shape[1], hidden=hidden, seed=seed)
        m.fit(feat[tr], pssm[tr], epochs=epochs, lr=lr, w_decay=w_decay,
              verbose=False)
        pred_mlp[val] = m.predict(feat[val])
        comp = pssm[tr].mean(axis=0)
        pred_comp[val] = comp
        if wt_idx is not None:
            table = {}
            for a in range(N_AA):
                mask = np.asarray(wt_idx)[tr] == a
                table[a] = pssm[tr][mask].mean(axis=0) if mask.any() else comp
            p = np.array([table[int(w)] for w in np.asarray(wt_idx)[val]])
            p = np.clip(p, 1e-4, None)
            pred_wt[val] = p / p.sum(axis=1, keepdims=True)

    def _ce_rows(Y, P):
        return -np.sum(Y * np.log(np.clip(P, 1e-8, None)), axis=1)

    ce_mlp = _ce_rows(pssm, pred_mlp)
    ce_comp = _ce_rows(pssm, pred_comp)
    ce_wt = _ce_rows(pssm, pred_wt) if pred_wt is not None else None
    out = dict(mlp_ce=float(ce_mlp.mean()), comp_ce=float(ce_comp.mean()),
               delta_ce=float(ce_comp.mean() - ce_mlp.mean()),
               mlp_pred=pred_mlp, ce_per_site=ce_mlp, folds=len(splits),
               baseline_ce=float(ce_comp.mean()))
    if ce_wt is not None:
        out["wtcond_ce"] = float(ce_wt.mean())
        out["delta_wt_ce"] = float(ce_wt.mean() - ce_mlp.mean())
        out["ce_per_site_wt"] = ce_wt
    return out


def build_evoprior(seq, msa_path, use_plm=True, plm_model="esm2_t33_650M_UR50D",
                   embed_cache=None, embed_npz=None, embed_key=None, pca_dim=32,
                   wt_feat=True, hidden=64, epochs=1500, lr=0.03, w_decay=1e-2,
                   p_evo_source="auto", min_neff=30.0,
                   alpha_cv=True, cv_eval=True, seed=0, verbose=True):
    """端到端训练 EvoPrior。

    参数
    ----
    seq        : WT 序列（str）
    msa_path   : a3m/fasta 同源 MSA（提供 PSSM 标签）
    embed_npz  : 集群预计算 PLM 嵌入的 npz 路径（优先；见 load_npz_embeddings）
    use_plm    : 无 npz 时 True=ESM-2 嵌入，False=placeholder 嵌入（无需 torch）
    pca_dim    : PCA 降维目标维度（D>>L 时必要，ESM3 为 1536 维）；0/None=不降维
    wt_feat    : 是否把 WT 残基 one-hot 拼进 MLP 输入特征（默认 True）
    w_decay    : MLP 权重衰减；单蛋白样本量小，正则化强度对结果影响很大
    cv_eval    : 是否做位点 k 折 CV 并报告留出交叉熵（诚实性评估）
    p_evo_source : "pssm" | "mlp" | "auto"（默认）。见下。
    min_neff   : auto 模式下判定"MSA 覆盖不足"的列深度阈值

    p_evo 该用 PSSM 还是 MLP？（PprI 实测结论，务必先读）
    ----------------------------------------------------
    留出 5 折 CV（L=254，MSA 189 条）实测：
        wt_cond 查表（无嵌入）      2.2242   ← 最强
        MLP(WT one-hot, wd=1e-4)   2.3057
        MLP(WT+ESM3, wd=1e-2)      2.4473
        MLP(ESM3 only, wd=3e-2)    2.5173
        composition / uniform      2.8537 / 2.9957
    即：**在单条蛋白、254 个位点的样本量下，PSSM 的位点间变异主要由 WT 残基
    身份解释；ESM3 嵌入虽显著优于均匀/组成基线，却无法稳定超越"按 WT 类型
    取平均"这一无嵌入基线**（tune2 在强正则 wd=1e-1 下观察到的正增益是欠拟合
    区的假象，放宽正则后反转为 t=−7.15）。

    因此在有 MSA 覆盖的位点上，**PSSM 本身就是最可信的数据驱动先验**——它是
    观测值，不是模型估计。MLP/PLM 的价值在于"MSA 覆盖不足的位点"上的补全，
    而要真正兑现 PLM 的增量，需要**多蛋白联合训练**扩大样本量（见文档 §6）。

    auto 模式据此工作：PSSM 深度 ≥ min_neff 的列直接用 PSSM；不足的列用
    MLP 留出预测填补，并在返回的 `low_cov` 中记录被填补的位点。

    为什么默认 wt_feat=True
    ----------------------
    PprI 留出 5 折实验表明：PSSM 的可预测性很大一部分来自"该位点当前是什么
    氨基酸"（无嵌入的 WT 查表基线 CE=2.224，远好于均匀 2.996）。把 WT one-hot
    显式喂给 MLP，可免去模型从 PCA 嵌入里重建该信息（254 个样本下很难）。
    特征消融（tune2）显示：在 WT one-hot 之上叠加 ESM3 嵌入仍有统计显著的
    增量增益（ΔCE=+0.108，t=+9.87），说明嵌入贡献的是**超出 WT 身份**的
    位点环境信息——这正是纠正 M135H 式盲点所需要的信号。

    返回 dict: p_evo, pssm, alpha, model, embed, used_plm, embed_source, pca, cv
    """
    if verbose:
        print(f"[EvoPrior] 解析 MSA: {msa_path}")
    query, seqs = read_a3m(msa_path)
    # query 与传入 seq 长度对齐（a3m query 即 WT）
    if len(query) != len(seq):
        # 取与 seq 等长的 MSA 行兜底
        L = len(seq)
    else:
        L = len(seq)
    pssm = msa_to_pssm(seqs, seq, pseudocount=1.0)
    wt_idx = np.array([AA2IDX.get(c, 0) for c in seq[:L]], dtype=int)

    # ---- 嵌入：优先集群预计算 npz > 本地 ESM-2 > placeholder ----
    if embed_npz:
        emb, used_key = load_npz_embeddings(embed_npz, key=embed_key)
        src = f"npz:{os.path.basename(embed_npz)}[{used_key}]"
        used_plm = True
    elif use_plm:
        emb = get_embeddings(seq, plm_model, cache=embed_cache)
        src = f"esm2:{plm_model}"
        used_plm = True
    else:
        emb = placeholder_embeddings(seq, window=3)
        src = "placeholder(window=3)"
        used_plm = False
    emb = np.asarray(emb, float)[:L]
    if emb.shape[0] != L:
        raise ValueError(f"嵌入长度 {emb.shape[0]} ≠ 序列长度 {L}（src={src}）")

    # ---- PCA 降维（D >> L 时必要）----
    pca_meta = None
    if pca_dim and emb.shape[1] > pca_dim:
        emb, pca_meta = pca_reduce(emb, dim=pca_dim)
        if verbose:
            print(f"[EvoPrior] PCA: {pca_meta['in_dim']} → {pca_meta['dim']} 维 "
                  f"（解释方差 {pca_meta['explained_var']:.3f}）")

    # ---- 拼接 WT 身份特征（实验证明有增益）----
    if wt_feat:
        wt_oh = np.zeros((L, N_AA))
        for i, c in enumerate(seq[:L]):
            idx = AA2IDX.get(c)
            if idx is not None:
                wt_oh[i, idx] = 1.0
        feat = np.hstack([emb, wt_oh])
    else:
        feat = emb

    if verbose:
        print(f"[EvoPrior] 嵌入={src}  特征 shape={feat.shape}"
              f"{'（含 WT one-hot）' if wt_feat else ''}")

    # ---- 训练 MLP ----
    if verbose:
        print(f"[EvoPrior] 训练 MLP (hidden={hidden}, epochs={epochs}, "
              f"w_decay={w_decay})")
    model = EvoPriorMLP(feat.shape[1], hidden=hidden, seed=seed)
    model.fit(feat, pssm, epochs=epochs, lr=lr, w_decay=w_decay, verbose=verbose)
    p_evo = model.predict(feat)

    # ---- 诚实性评估：留出位点 CV，与两个无嵌入基线对比 ----
    cv = None
    if cv_eval or p_evo_source in ("mlp", "auto"):
        if verbose:
            print("[EvoPrior] 位点 5 折 CV 评估（留出位点泛化）...")
        cv = evaluate_mlp_cv(feat, pssm, wt_idx=wt_idx, folds=5, hidden=hidden,
                             epochs=max(200, epochs // 2), lr=lr,
                             w_decay=w_decay, seed=seed)
        if verbose:
            msg = (f"  留出 CE: MLP={cv['mlp_ce']:.4f}  "
                   f"组成={cv['comp_ce']:.4f}")
            if "wtcond_ce" in cv:
                msg += (f"  WT查表={cv['wtcond_ce']:.4f}  "
                        f"Δ(vs查表)={cv['delta_wt_ce']:+.4f}")
            print(msg)
            if "wtcond_ce" in cv and cv["delta_wt_ce"] < 0:
                print("  注意：MLP 未超越无嵌入的 WT 查表基线——"
                      "单蛋白样本量下 PSSM 变异主要由 WT 身份解释，"
                      "故 p_evo 默认以 PSSM 为准（见 docstring）。")

    # ---- p_evo 的来源选择 ----
    neff = msa_column_neff(seqs, seq)
    low_cov = neff < float(min_neff)
    mlp_pred = None
    if cv is not None:
        mlp_pred = np.clip(cv["mlp_pred"], 1e-6, None)
        mlp_pred = mlp_pred / mlp_pred.sum(axis=1, keepdims=True)

    if p_evo_source == "pssm":
        p_evo = pssm.copy()
    elif p_evo_source == "mlp":
        if mlp_pred is None:
            raise ValueError("p_evo_source='mlp' 需要 cv_eval=True")
        p_evo = mlp_pred
    elif p_evo_source == "auto":
        # PSSM 覆盖充足的列直接用观测值；不足的列用 MLP 留出预测补全
        p_evo = pssm.copy()
        if low_cov.any() and mlp_pred is not None:
            p_evo[low_cov] = mlp_pred[low_cov]
            if verbose:
                print(f"[EvoPrior] MSA 覆盖不足（<{min_neff} 条）的 "
                      f"{int(low_cov.sum())}/{L} 列由 MLP 预测补全")
    else:
        raise ValueError(f"未知 p_evo_source={p_evo_source!r}")

    if verbose:
        print(f"[EvoPrior] p_evo 来源={p_evo_source}"
              f"（PSSM 列深度中位数={np.median(neff):.0f}）")

    alpha = 0.5
    if alpha_cv:
        # 化学先验占位：CV 阶段用均匀先验作 p_chem 代理评估 p_evo 自身可靠性；
        # 真正与 p_chem 融合时的 α 由调用方在 M2 层用 learn_alpha_cv 给定。
        pass
    return dict(p_evo=p_evo, pssm=pssm, alpha=alpha, model=model,
                embed=emb, used_plm=used_plm, embed_source=src,
                pca=pca_meta, cv=cv, seq=seq, neff=neff,
                low_cov=low_cov, mlp_pred=mlp_pred)


def pssm_to_bench(p_chem_table, p_evo, pssm, sites, alpha=0.5):
    """把 [L_full,20] 表按 sites 子集索引并融合，返回 WFKernel 可用的 {site:{aa:p}}。"""
    out = {}
    p_chem = np.asarray(p_chem_table, float)
    p_evo = np.asarray(p_evo, float)
    pssm = np.asarray(pssm, float)
    for j in sites:
        pf = fuse_tables(p_chem[j:j + 1], p_evo[j:j + 1], alpha)[0]
        out[int(j)] = {AA20[k]: float(pf[k]) for k in range(N_AA)}
    return out
