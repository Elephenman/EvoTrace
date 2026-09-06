# -*- coding: utf-8 -*-
"""事件脚本 + 长时程驱动器（v1 方案 §3.4"事件脚本"，路线图 R2）。

支持的事件类型（与 envspec 预设 dynamics.events 的 schema 对齐）：
  bottleneck    种群瓶颈: Ne 骤降 factor 倍, 持续 n 代
  env_ramp      环境渐变（盐/温度等环境轴）: 在 n 代内把先验表从 start 线性插值到 end
                （经 kernel.set_prior_table, 即 M4 回流同一入口——环境变化=景观变化）
  salt_ramp     env_ramp 的盐特化别名（预设 schema 兼容）
  migration     基因流/HGT: 每代把 fraction 比例个体替换为供体基因型
  temperature_shift  选择温度台阶: kernel.T 改值（软化/收紧选择强度）

驱动器 run_with_events 逐代调用 kernel.run（每种群独立续跑, 保持平行种群独立性）,
每代结束后应用当代事件并回调 observer(gen, pops, stats)——谱系记录(lineage.py)
与缓存统计经此挂入, kernel 本身零改动。
时间轴: EnvSpec 以年记事件, 经 generations_per_year 换算为代; 年轴三情景换算由
lineage.report 层输出（v1 §3.5）。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = ["EventScript", "run_with_events"]

# 预设/EnvSpec 事件 schema → 引擎事件参数
_TYPE_ALIASES = {"salt_ramp": "env_ramp",
                 "event_salt_ramp": "env_ramp",
                 "event_bottleneck": "bottleneck",
                 "event_migration": "migration",
                 "event_temperature_shift": "temperature_shift"}


class Event:
    def __init__(self, type_, t_gen, duration_g, **params):
        self.type = _TYPE_ALIASES.get(type_, type_)
        self.t_gen = int(t_gen)
        self.duration_g = int(duration_g)
        self.params = params

    def active(self, gen: int) -> bool:
        return self.t_gen <= gen < self.t_gen + self.duration_g

    def fraction(self, gen: int) -> float:
        """env_ramp 进度 0→1。"""
        if self.duration_g <= 0:
            return 1.0
        return min(1.0, max(0.0, (gen - self.t_gen) / self.duration_g))

    def __repr__(self):
        return f"<Event {self.type} @gen{self.t_gen} +{self.duration_g}g {self.params}>"


class EventScript:
    """事件列表。from_envspec 把 dynamics.events（年轴）换算为代轴。"""

    def __init__(self, events: List[Event]):
        self.events = sorted(events, key=lambda e: e.t_gen)

    @classmethod
    def from_envspec(cls, dynamics: Dict) -> "EventScript":
        gpy = float(dynamics.get("generations_per_year", 300) or 300)
        evs = []
        for raw in dynamics.get("events", []) or []:
            t = raw.get("t_year", raw.get("t_gen", 0))
            dur = raw.get("duration_y", raw.get("duration_g", 0))
            evs.append(Event(raw["type"], int(t * gpy), int(dur * gpy), **{
                k: v for k, v in raw.items()
                if k not in ("t_year", "t_gen", "duration_y", "duration_g", "type")}))
        return cls(evs)

    def events_at(self, gen: int) -> List[Event]:
        return [e for e in self.events if e.active(gen)]

    def __repr__(self):
        return f"<EventScript {self.events}>"


def _interp_prior_table(a: np.ndarray, b: np.ndarray, frac: float) -> np.ndarray:
    t = a * (1.0 - frac) + b * frac
    t = np.clip(t, 1e-4, None)
    return t / t.sum(axis=1, keepdims=True)


def run_with_events(kernel, script: EventScript, n_pop: int = 8, n_gen: int = 1000,
                    Ne: int = 500, pops0: Optional[np.ndarray] = None,
                    observer: Optional[Callable[[int, np.ndarray, List[dict]], None]] = None):
    """带事件脚本的长时程驱动器。返回 (all_stats, pops, event_log)。

    每代每种群独立 kernel.run(n_pop=1, n_gen=1)（保持平行种群谱系不串）;
    事件在代边界应用; observer 每代收到 (gen, pops[n_pop,Ne,L], 该代 stats)。
    """
    if pops0 is None:
        pops = np.tile(kernel.wt_idx[None, None, :], (n_pop, Ne, 1))
    else:
        pops = np.array(pops0, dtype=np.int8)
        n_pop, Ne = pops.shape[0], pops.shape[1]
    event_log: List[dict] = []
    all_stats: List[dict] = []
    # env_ramp 的起点表固定为事件触发时的表（渐变从"当时环境"出发）
    ramp_start: Dict[int, np.ndarray] = {}
    for g in range(n_gen):
        cur_ne, cur_t = Ne, None
        for ev in script.events_at(g):
            if ev.type == "bottleneck":
                cur_ne = max(2, int(Ne * float(ev.params.get("factor", 0.1))))
            elif ev.type == "env_ramp":
                if ev.t_gen not in ramp_start:
                    ramp_start[ev.t_gen] = kernel.prior_table.copy()
                kernel.set_prior_table(_interp_prior_table(
                    ramp_start[ev.t_gen], np.asarray(ev.params["end_table"], float),
                    ev.fraction(g)))
            elif ev.type == "temperature_shift":
                cur_t = float(ev.params["to_T"])
            elif ev.type == "migration":
                donor = ev.params.get("donor")          # None → WT
                frac = float(ev.params.get("fraction", 0.05))
                geno = (kernel.wt_idx if donor is None
                        else np.asarray(donor, dtype=np.int8))
                n_swap = max(1, int(Ne * frac))
                for p in range(n_pop):
                    idx = kernel.rng.choice(Ne, size=n_swap, replace=False)
                    pops[p, idx] = geno
            event_log.append(dict(gen=g, type=ev.type, **{
                k: v for k, v in ev.params.items() if k != "end_table"}))
        if cur_t is not None:
            kernel.T = cur_t
        gen_stats: List[dict] = []
        for p in range(n_pop):
            # 瓶颈: 抽 cur_ne 个体为奠基者; 恢复: 复制回 Ne
            founders = pops[p][kernel.rng.choice(Ne, size=cur_ne, replace=False)]
            if cur_ne != Ne:
                rep = kernel.rng.choice(cur_ne, size=Ne - cur_ne)
                founders = np.vstack([founders, founders[rep]])
            else:
                founders = pops[p]
            _, newpop = kernel.run(n_pop=1, n_gen=1, Ne=Ne, founder=founders,
                                   record_events=False)
            pops[p] = newpop
            geno = pops[p]
            fits = kernel._fitness(geno)
            gen_stats.append(dict(pop=p, gen=g,
                                  best=round(float(fits.max()), 4),
                                  mean=round(float(fits.mean()), 4),
                                  unique=int(len(np.unique(geno, axis=0)))))
        all_stats.extend(gen_stats)
        if observer is not None:
            observer(g, pops, gen_stats)
    return all_stats, pops, event_log
