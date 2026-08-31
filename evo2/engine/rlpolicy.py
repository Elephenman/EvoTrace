#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — RL 分支：factored DQN 在线策略 + 离线行为克隆阴性对照。

设计（对应"方案一：RL 替代 Wright-Fisher 内核"）：
  * 状态: 当前基因型（53 位点）+ 突变负荷 + 当前适应度（相对 WT 归一）。
  * 动作: (位点 j, 氨基酸 a) 的 factored 空间 L×20；共享 MLP Q 网络，
    动作特征 = [位点嵌入 E[j], 当前 AA one-hot, 目标 AA one-hot, 上下文]。
  * 奖励: r_t = f_{t+1} − f_t（oracle 增量整形），γ<1 折扣累计 ≈ 终态适应度。
  * 奖励来源: oracle（按评估预算计费），**不**在线调 Boltz-2 —— 3,240 行
    per_model_metrics 只有 36 条独立序列，撑不起在线 GPU 评估（硬规则 §9.1）。
  * 训练: DQN（replay + target net + ε-greedy），纯 numpy（无 torch 依赖）。
  * BCOptimizer: 离线行为克隆阴性对照 —— 用 36 条真实 PprI 系统的突变列表
    做监督克隆，检验"36 条样本在 1060 维动作空间上记忆化"的失败程度。

诚实性说明：36 条真实序列与加性代理零相关（corr(add_fit, sep)=-0.11），
BC 的训练信号本身不携带可靠景观信息，其阴性结果是预期结论而非意外。
"""
import os

import numpy as np

from .wfopt import BaseOptimizer, register


# ----------------------------------------------------------------------
class _MLP:
    """单隐层标量输出 MLP + 学习的位点嵌入表，纯 numpy。"""

    def __init__(self, in_dim, hidden, n_sites, d_embed, rng, lr=1e-3):
        self.lr = lr
        self.E = rng.normal(0, 0.1, size=(n_sites, d_embed))
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), size=(hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X, E_rows=None):
        """X [B, in]; E_rows [B] 或 None（X 已含嵌入）。返回 y [B], cache。"""
        self._last_rows = E_rows
        if E_rows is not None:
            X = np.concatenate([self.E[E_rows], X], axis=1)
        h = np.maximum(X @ self.W1 + self.b1, 0.0)
        y = (h @ self.W2 + self.b2).ravel()
        return y, (X, h)

    def backward(self, cache, dy):
        X, h = cache
        dy = np.asarray(dy).reshape(-1, 1)
        dW2 = h.T @ dy / len(X)
        db2 = dy.mean(axis=0)
        dh = dy @ self.W2.T
        dh[h <= 0] = 0.0
        dX = dh @ self.W1.T                      # [B, in_full]
        dW1 = X.T @ dh / len(X)
        db1 = dh.mean(axis=0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        # 嵌入梯度回传（scatter-add 到对应行）
        if self._last_rows is not None:
            dE = dX[:, :self.E.shape[1]]
            np.add.at(self.E, self._last_rows, -self.lr * dE)
        return dW1, db1

    def state_dict(self):
        return {k: (v.copy() if hasattr(v, "copy") else v)
                for k, v in self.__dict__.items()
                if k not in ("_last_rows",)}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if hasattr(v, "copy"):
                setattr(self, k, v.copy())
            else:
                setattr(self, k, v)


class DQNOptimizer(BaseOptimizer):
    """Factored DQN：每步一个 (site, aa) 编辑，episode 最长 max_mut 步。"""

    name = "dqn"

    def __init__(self, oracle, seed=0, budget=4000, cfg=None):
        super().__init__(oracle, seed, budget, cfg)
        c = self.cfg
        self.gamma = float(c.get("gamma", 0.9))
        self.eps0, self.eps1 = 1.0, float(c.get("eps_min", 0.05))
        self.eps_decay = int(c.get("eps_decay_steps", 2000))
        self.batch = int(c.get("batch", 32))
        self.buf_size = int(c.get("buf_size", 20000))
        self.hidden = int(c.get("hidden", 64))
        self.d_embed = int(c.get("d_embed", 8))
        self.max_steps = int(c.get("max_steps", oracle.max_mut or 12))
        self.train_every = int(c.get("train_every", 1))
        self.rng_net = np.random.default_rng(seed + 1)
        in_dim = self.d_embed + 20 + 20 + 2
        self.net = _MLP(in_dim, self.hidden, oracle.L, self.d_embed,
                        self.rng_net, lr=float(c.get("lr", 1e-3)))
        self.target = _MLP(in_dim, self.hidden, oracle.L, self.d_embed,
                           self.rng_net, lr=0.0)
        self.target.load_state_dict(self.net.state_dict())
        self.buf = []          # (geno_prev[L], f_prev, j, a, reward, geno_next[L], done)
        self.train_steps = 0
        self.env_steps = 0
        self._f_wt = None

    # ---------- 特征 ----------
    def _action_features(self, geno, f_cur):
        """返回 X [L*20, 42]（不含嵌入，嵌入由 forward 的 E_rows 提供）
        与 E_rows [L*20]。"""
        L = self.oracle.L
        cur = geno[:, None] == np.arange(20)[None, :]          # [L,20]
        cur_t = np.repeat(cur, 20, axis=0)                     # [L*20,20] (j,a), a 最快
        aa_t = np.tile(np.eye(20), (L, 1))                     # [L*20,20] 目标AA one-hot
        n_mut = int((geno != self.oracle.wt_idx).sum())
        ctx1 = n_mut / max(self.max_steps, 1)
        ctx2 = float(np.tanh(f_cur - (self._f_wt or 0.0)))
        ctx = np.tile([ctx1, ctx2], (L * 20, 1))
        X = np.concatenate([cur_t, aa_t, ctx], axis=1)
        E_rows = np.repeat(np.arange(L), 20)
        return X, E_rows

    def _q_all(self, geno, f_cur, net):
        X, E_rows = self._action_features(geno, f_cur)
        q, _ = net.forward(X, E_rows)
        return q.reshape(self.oracle.L, 20), (X, E_rows)

    # ---------- DQN 训练 ----------
    def _train_step(self):
        if len(self.buf) < self.batch:
            return
        idx = self.rng_net.integers(0, len(self.buf), size=self.batch)
        batch = [self.buf[i] for i in idx]
        # 当前 Q(s, j, a)（特征与采集时完全一致：同一 f_prev）
        Xs, Ers = [], []
        for geno, f_prev, j, a, r, g2, done in batch:
            X, E_rows = self._action_features(geno, f_prev)
            Xs.append(X[j * 20 + a])
            Ers.append(E_rows[j * 20 + a])
        Xs = np.array(Xs)
        q, cache = self.net.forward(Xs, np.array(Ers))
        # target: r + γ max_a' Q_target(s')
        tgt = np.zeros(len(batch))
        for i, (geno, f_prev, j, a, r, g2, done) in enumerate(batch):
            if done:
                tgt[i] = r
            else:
                q2, _ = self._q_all(g2, r + f_prev, self.target)
                q2m = q2.copy()
                q2m[np.arange(self.oracle.L), g2] = -np.inf
                tgt[i] = r + self.gamma * float(np.max(q2m))
        dy = 2.0 * (q - tgt) / len(batch)
        self.net.backward(cache, dy)
        self.train_steps += 1
        if self.train_steps % 200 == 0:
            self.target.load_state_dict(self.net.state_dict())

    # ---------- 主循环 ----------
    def run(self):
        o = self.oracle
        self._f_wt = float(o.evaluate(o.wt_idx[None, :])[0])   # 计 1 eval
        self._observe(o.wt_idx[None, :], np.array([self._f_wt]))
        L = o.L
        while not self._done():
            geno = o.wt_idx.copy()
            f_cur = self._f_wt
            eps = max(self.eps1, self.eps0 - self.eps0 * self.env_steps / self.eps_decay)
            for step in range(self.max_steps):
                if self._done():
                    break
                q, _ = self._q_all(geno, f_cur, self.net)
                q_mask = q.copy()
                q_mask[np.arange(L), geno] = -np.inf      # 无操作动作屏蔽
                if self.rng.random() < eps:
                    j = int(self.rng.integers(L))
                    aa = int(self.rng.integers(20))
                    if aa == geno[j]:
                        continue
                else:
                    flat = int(np.nanargmax(q_mask))
                    j, aa = divmod(flat, 20)
                g2 = geno.copy()
                g2[j] = aa
                f2 = float(o.evaluate(g2[None, :])[0])    # 计 1 eval
                r = f2 - f_cur
                done = (step == self.max_steps - 1)
                self.buf.append((geno.copy(), f_cur, j, aa, r, g2.copy(), done))
                if len(self.buf) > self.buf_size:
                    self.buf.pop(0)
                self._observe(g2[None, :], np.array([f2]))
                geno, f_cur = g2, f2
                self.env_steps += 1
                if self.env_steps % self.train_every == 0:
                    self._train_step()
        return dict(best_f=self.best_f, best_geno=self.best_g,
                    trace=self.trace, n_evals=o.n_evals)


# ----------------------------------------------------------------------
class BCOptimizer(BaseOptimizer):
    """离线行为克隆阴性对照。

    训练数据：36 条真实 PprI 系统（A:/claudework/out/ppri_real_dataset.csv，
    用户数据不入库，文件缺失时跳过并说明）。监督目标 = 专家 (site,aa) 的
    softmax 交叉熵；状态 = 按位置顺序施加前缀突变后的基因型。
    评估阶段：贪心跟随克隆策略在 oracle 上走 12 步（正常计预算）。
    """

    name = "bc"
    DATA = "A:/claudework/out/ppri_real_dataset.csv"

    def __init__(self, oracle, seed=0, budget=4000, cfg=None):
        super().__init__(oracle, seed, budget, cfg)
        c = self.cfg
        self.hidden = int(c.get("hidden", 64))
        self.d_embed = int(c.get("d_embed", 8))
        self.epochs = int(c.get("epochs", 200))
        self.rng_net = np.random.default_rng(seed + 2)
        in_dim = self.d_embed + 20 + 20 + 2
        self.net = _MLP(in_dim, self.hidden, oracle.L, self.d_embed,
                        self.rng_net, lr=float(c.get("lr", 3e-3)))
        self._f_wt = None
        self.max_steps = int(c.get("max_steps", oracle.max_mut or 12))
        self.traj = self._load_trajectories()
        self.bc_loss = self._train_bc() if self.traj else float("nan")

    def _load_trajectories(self):
        if not os.path.exists(self.DATA):
            return []
        import pandas as pd
        from .seqtools import AA2IDX
        df = pd.read_csv(self.DATA)
        trajs = []
        for muts in df["muts"]:
            acts = []
            for tok in str(muts).split(";"):
                if not tok:
                    continue
                digits = "".join(ch for ch in tok if ch.isdigit())
                aa = tok[-1]
                pos = int(digits) - 22          # PDB offset
                js = np.flatnonzero(self.oracle.sites == pos)
                if len(js) and aa in AA2IDX:
                    acts.append((int(js[0]), AA2IDX[aa]))
            if acts:
                trajs.append(acts)
        return trajs

    def _train_bc(self):
        if not self.traj:
            return float("nan")
        states = []          # (geno_prefix, expert_flat_action)
        for acts in self.traj:
            geno = self.oracle.wt_idx.copy()
            for j, a in acts:
                states.append((geno.copy(), j * 20 + a))
                geno[j] = a
        losses = []
        bs = 8    # 状态数/批；每个状态展开 L*20 个动作行
        for ep in range(self.epochs):
            idx = self.rng_net.integers(0, len(states), size=min(bs, len(states)))
            Xb, Eb, tb = [], [], []
            for i in idx:
                geno, tgt = states[i]
                X, E_rows = self._features(geno)
                Xb.append(X)
                Eb.append(E_rows)
                tb.append(tgt)
            # 每个状态是 [L*20, in_dim] 块 → 沿动作维展平为 [B*L*20, in_dim]
            Xb = np.concatenate(Xb, axis=0)
            Eb = np.concatenate(Eb, axis=0)
            tb = np.array(tb)
            B = len(tb)
            q, cache = self.net.forward(Xb, Eb)      # [B*L*20]
            q = q.reshape(B, -1)                     # [B, L*20]
            z = q - q.max(axis=1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            losses.extend((-np.log(np.maximum(p[np.arange(B), tb], 1e-12))).tolist())
            dy = p.copy()
            dy[np.arange(B), tb] -= 1.0
            # softmax-CE 对逐行 logit 的梯度：目标行 p-1，其余行 p；
            # 不再额外除 B —— backward 内部按 len(X)=B*L*20 归一。
            self.net.backward(cache, dy.reshape(-1))
        return float(np.mean(losses[-50:]))

    def _features(self, geno):
        L = self.oracle.L
        cur = geno[:, None] == np.arange(20)[None, :]
        cur_t = np.repeat(cur, 20, axis=0)
        aa_t = np.tile(np.eye(20), (L, 1))
        n_mut = int((geno != self.oracle.wt_idx).sum())
        ctx = np.tile([n_mut / max(self.oracle.max_mut or 12, 1), 0.0], (L * 20, 1))
        return np.concatenate([cur_t, aa_t, ctx], axis=1), np.repeat(np.arange(L), 20)

    def run(self):
        o = self.oracle
        if not self.traj:
            return dict(best_f=float("nan"), best_geno=None, trace=[],
                        n_evals=0, note="BC 数据缺失，跳过")
        self._f_wt = float(o.evaluate(o.wt_idx[None, :])[0])
        L = o.L
        geno = o.wt_idx.copy()
        for step in range(self.max_steps):
            if self._done():
                break
            X, E_rows = self._features(geno)
            q, _ = self.net.forward(X, E_rows)
            q = q.reshape(L, 20)
            q[np.arange(L), geno] = -np.inf
            j, a = divmod(int(np.argmax(q)), 20)
            geno[j] = a
            f = float(o.evaluate(geno[None, :])[0])
            self._observe(geno[None, :], np.array([f]))
        return dict(best_f=self.best_f, best_geno=self.best_g, trace=self.trace,
                    n_evals=o.n_evals, bc_loss=round(self.bc_loss, 4))
