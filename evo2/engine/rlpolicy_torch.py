# -*- coding: utf-8 -*-
"""torch 深度学习策略：PPO (actor-critic) 进化内核 —— rl-policy 分支的 DL 升级版。

与 rlpolicy.py 的 numpy DQN 同接口（BaseOptimizer / OPTIMIZERS 注册表），
但网络与训练全部走 torch：
- 状态  = 序列 one-hot (L*20) + [归一化当前适应度, 突变预算余量] → 2 层 MLP trunk
- 动作  = L*20 flat（(site, aa) 对），no-op 与超 max_mut 的动作被 mask 为 -inf
- 奖励  = oracle 适应度增量 Δf（用 ref-wt 归一化），episode 终止于 max_mut 或预算
- 训练  = PPO-clip + GAE(λ) + 熵正则，rollout 2048 步一更新，minibatch 梯度步
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.wfopt import BaseOptimizer, register

AA = 20


class _TorchNet(nn.Module):
    """共享 trunk + policy/value 双头。"""

    def __init__(self, obs_dim, n_act, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_act)
        self.v = nn.Linear(hidden, 1)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0); nn.init.zeros_(self.v.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


@register
class PPOOptimizer(BaseOptimizer):
    name = "ppo"

    def __init__(self, oracle, seed=0, budget=4000, cfg=None):
        super().__init__(oracle, seed=seed, budget=budget, cfg=cfg)
        c = self.cfg
        self.max_steps = int(c.get("max_steps", oracle.max_mut or 12))
        self.device = "cpu"
        torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        L = oracle.L
        self.n_act = L * AA
        self.obs_dim = L * AA + 2
        self.net = _TorchNet(self.obs_dim, self.n_act).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(c.get("lr", 3e-4)))
        self.gamma = float(c.get("gamma", 0.99))
        self.lam = float(c.get("lam", 0.95))
        self.clip = float(c.get("clip", 0.2))
        self.ent = float(c.get("ent", 0.01))
        self.rollout = int(c.get("rollout", 2048))
        self.epochs = int(c.get("epochs", 4))
        self.mb = int(c.get("mb", 256))
        # 每个 genotype 已评估一次的适应度缓存（避免重复计预算）
        self._fcache = {}
        # 归一化尺度：WT 与一步上限（用 oracle 提供的 ref 若有）
        self._f0 = float(oracle.evaluate(oracle.wt_idx[None, :])[0])
        self._scale = 1.0
        ref = getattr(oracle, "known_optimum", lambda: None)()
        if ref is not None:
            # known_optimum 可能返回 geno 或 (fitness, geno)
            fr = float(ref[0]) if isinstance(ref, tuple) else float(
                oracle.evaluate(np.asarray(ref)[None, :])[0])
            if fr > self._f0:
                self._scale = fr - self._f0

    # ---------- 状态 / 动作 ----------
    def _obs(self, geno, f_cur):
        x = np.zeros(self.obs_dim, dtype=np.float32)
        x[geno + AA * np.arange(len(geno))] = 1.0
        x[-2] = (f_cur - self._f0) / max(self._scale, 1e-6)
        x[-1] = 1.0 - (int((geno != self.oracle.wt_idx).sum()) / max(self.max_steps, 1))
        return x

    def _mask(self, geno):
        m = np.zeros(self.n_act, dtype=bool)
        n_mut = int((geno != self.oracle.wt_idx).sum())
        for j in range(self.oracle.L):
            if geno[j] == self.oracle.wt_idx[j] and n_mut >= self.max_steps:
                continue  # 已满额不能再加突变
            base = j * AA
            m[base + geno[j]] = True        # no-op 屏蔽
            if geno[j] == self.oracle.wt_idx[j] and n_mut >= self.max_steps:
                m[base:base + AA] = True    # 双保险
        return ~m

    def _step(self, geno, f_cur):
        """单个环境步：采样一个动作并评估。返回 (a, logp, v, geno2, f2, r, done)。"""
        obs = torch.tensor(self._obs(geno, f_cur)).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.net(obs)
        mask = torch.tensor(self._mask(geno))
        logits = logits.squeeze(0).masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        j, aa = divmod(int(a), AA)
        geno2 = geno.copy()
        geno2[j] = aa
        key = geno2.tobytes()
        if key not in self._fcache:
            f2 = float(self.oracle.evaluate(geno2[None, :])[0])
            self._fcache[key] = f2
        f2 = self._fcache[key]
        self._observe(geno2[None, :], np.array([f2]))
        r = (f2 - f_cur) / max(self._scale, 1e-6)
        done = self._done() or int((geno2 != self.oracle.wt_idx).sum()) >= self.max_steps
        return int(a), float(logp), float(v), geno2, f2, r, done

    # ---------- PPO 更新 ----------
    def _update(self, buf):
        obs = torch.tensor(np.array([b[0] for b in buf]))
        act = torch.tensor([b[1] for b in buf])
        old_logp = torch.tensor([b[2] for b in buf])
        rew = torch.tensor([b[3] for b in buf], dtype=torch.float32)
        done = torch.tensor([b[4] for b in buf], dtype=torch.float32)
        val = torch.tensor([b[5] for b in buf], dtype=torch.float32)
        # GAE
        T = len(buf)
        adv = torch.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            nxt = 0.0 if done[t] else val[t + 1] if t + 1 < T else 0.0
            delta = rew[t] + self.gamma * nxt - val[t]
            last = delta + self.gamma * self.lam * (0.0 if done[t] else last)
            adv[t] = last
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(T)
        for _ in range(self.epochs):
            self.np_rng.shuffle(idx)
            for s in range(0, T, self.mb):
                i = torch.tensor(idx[s:s + self.mb])
                logits, v = self.net(obs[i])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[i])
                ratio = torch.exp(logp - old_logp[i])
                a_i = adv[i]
                pl = -torch.min(ratio * a_i, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_i).mean()
                vl = F.mse_loss(v, ret[i])
                ent = dist.entropy().mean()
                loss = pl + 0.5 * vl - self.ent * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

    def run(self):
        while not self._done():
            buf = []
            # 并行收集 rollout_steps 个环境步（多 episode 混合）
            n_start = self.oracle.n_evals
            genos = [self.oracle.wt_idx.copy()]
            fs = [self._f0]
            while self.oracle.n_evals - n_start < min(self.rollout, self.budget - n_start):
                g, f = genos[-1], fs[-1]
                a, logp, v, g2, f2, r, done = self._step(g, f)
                buf.append((self._obs(g, f), a, logp, r, float(done), v))
                if done:
                    genos.append(self.oracle.wt_idx.copy())
                    fs.append(self._f0)
                else:
                    genos.append(g2)
                    fs.append(f2)
            if buf:
                self._update(buf)
        return {"best_f": self.best_f, "best_g": self.best_g,
                "trace": self.trace, "n_evals": self.oracle.n_evals}
