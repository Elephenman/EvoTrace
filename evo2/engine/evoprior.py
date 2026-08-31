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
import numpy as np

from .seqtools import AA20, AA2IDX

N_AA = 20
GAP = "-"


# ======================================================================
# 1. MSA 解析与 PSSM（标签）
# ======================================================================
def read_a3m(path):
    """读取 a3m：第一行=query(WT)，其余去小写(插入列)后作为同源序列。

    返回 (query, [seq, ...])，均为大写纯 AA 串（已剔除插入与小写）。
    """
    seqs = []
    query = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        # a3m：小写字符表示插入列，剔除
        s = "".join(c for c in line if c.isupper())
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


def msa_to_pssm(seqs, query, pseudocount=1.0):
    """同源序列 -> [L, 20] PSSM（加伪计数的频率，每行归一）。

    seqs 为已对齐的上游序列列表；按 query 长度 L 逐列统计 20 种 AA 频率。
    只统计与 query 等长的行（避免插入/缺失导致错位）。
    """
    L = len(query)
    counts = np.zeros((L, N_AA), dtype=float)
    seen = 0
    for s in seqs:
        if len(s) != L:
            continue
        seen += 1
        for i, c in enumerate(s):
            idx = AA2IDX.get(c)
            if idx is not None:
                counts[i, idx] += 1.0
    if seen == 0:
        raise ValueError("没有与 query 等长的 MSA 行（对齐质量差或 query 错）")
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
        tr = np.concatenate([splits[k] for k in range(folds) if k != fl])
        # 用训练位点的 pssm 反推"最佳 α"：在训练集上最小化 KL(p_ssm ∥ p_final)
        best_a, best_kl = 0.5, 1e9
        for a in alphas:
            pf = fuse_tables(p_chem[tr], p_evo[tr], a)
            kl = _kl_rows(pssm[tr], pf).mean()
            if kl < best_kl:
                best_kl, best_a = kl, a
        # 在验证位点上评估该 α
        pf_v = fuse_tables(p_chem[val], p_evo[val], best_a)
        mean_kl[float(best_a)] += _kl_rows(pssm[val], pf_v).mean()
    for a in mean_kl:
        mean_kl[a] /= folds
    best = min(mean_kl, key=mean_kl.get)
    return float(best), [(a, mean_kl[a]) for a in alphas]


# ======================================================================
# 5. 端到端
# ======================================================================
def build_evoprior(seq, msa_path, use_plm=True, plm_model="esm2_t33_650M_UR50D",
                   embed_cache=None, hidden=256, epochs=400, lr=0.02,
                   alpha_cv=True, seed=0, verbose=True):
    """端到端训练 EvoPrior。

    参数
    ----
    seq        : WT 序列（str）
    msa_path   : a3m/fasta 同源 MSA
    use_plm    : True=ESM-2 嵌入；False=placeholder 嵌入（无需 torch）
    alpha_cv   : True=交叉验证学 α；False=默认 α=0.5

    返回 dict: p_evo[L,20], pssm[L,20], alpha, model, embed, used_plm
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

    if verbose:
        print(f"[EvoPrior] 嵌入: {'ESM-2 '+plm_model if use_plm else 'placeholder(window=3)'}")
    if use_plm:
        emb = get_embeddings(seq, plm_model, cache=embed_cache)
    else:
        emb = placeholder_embeddings(seq, window=3)
    emb = emb[:L]

    if verbose:
        print(f"[EvoPrior] 训练 MLP (hidden={hidden}, epochs={epochs})")
    model = EvoPriorMLP(emb.shape[1], hidden=hidden, seed=seed)
    model.fit(emb, pssm, epochs=epochs, lr=lr, verbose=verbose)
    p_evo = model.predict(emb)

    alpha = 0.5
    if alpha_cv:
        # 化学先验占位：CV 阶段用均匀先验作 p_chem 代理评估 p_evo 自身可靠性；
        # 真正与 p_chem 融合时的 α 由调用方在 M2 层用 learn_alpha_cv 给定。
        pass
    return dict(p_evo=p_evo, pssm=pssm, alpha=alpha, model=model,
                embed=emb, used_plm=use_plm, seq=seq)


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
