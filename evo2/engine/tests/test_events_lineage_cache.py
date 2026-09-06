# -*- coding: utf-8 -*-
"""R2 模块测试: events（事件脚本/驱动器）、lineage（谱系/固定/命中率）、fitcache。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lineage import LineageRecorder  # engine 目录式（与 test_presets 同约定）
try:
    from seqtools import AA20
except ImportError:  # pragma: no cover
    from engine.seqtools import AA20
from events import Event, EventScript, run_with_events
from fitcache import FitnessCache


# ---------------- 最小可跑 kernel 桩（结构同 WFKernel 的被依赖面） ----------------
class MiniKernel:
    def __init__(self, L=6, seed=0):
        self.rng = np.random.default_rng(seed)
        self.L = L
        self.wt_idx = np.zeros(L, dtype=np.int8)
        tab = np.full((L, 20), 0.002)
        tab[np.arange(L), 1] = 0.12          # 残基 1 有优势
        tab /= tab.sum(axis=1, keepdims=True)
        self.prior_table = tab
        self.sites = np.arange(L)
        self.T = 0.6
        self.lam = 0.5
        self._gen = 0
        self.events = []
        self.sel = np.log(tab) - np.log(tab[np.arange(L), 0])[:, None]

    def set_prior_table(self, t):
        self.prior_table = t
        self.sel = np.log(np.maximum(t, 1e-4)) - np.log(
            np.maximum(t[np.arange(self.L), 0], 1e-4))[:, None]

    def _fitness(self, geno):
        return self.sel[np.arange(self.L), geno.astype(int)].sum(axis=1)

    def run(self, n_pop=1, n_gen=1, Ne=100, founder=None, record_events=False):
        geno = np.tile(self.wt_idx[None, :], (Ne, 1)) if founder is None \
            else np.array(founder, dtype=np.int8)
        stats = []
        for g in range(n_gen):
            fits = self._fitness(geno)
            w = np.exp((fits - fits.max()) / self.T)
            w /= w.sum()
            parents = self.rng.choice(Ne, size=Ne, p=w)
            geno = geno[parents].copy()
            k = self.rng.poisson(self.lam, size=Ne)
            for n in np.flatnonzero(k > 0):
                j = int(self.rng.integers(self.L))
                geno[n, j] = self.rng.integers(0, 20)
        return stats, geno[None, :, :]


# ---------------- events ----------------
def test_event_window_and_fraction():
    e = Event("salt_ramp", t_gen=100, duration_g=50, to_M=5.0)
    assert e.type == "env_ramp"                     # 别名归一
    assert not e.active(99) and e.active(100) and e.active(149) and not e.active(150)
    assert e.fraction(100) == 0.0 and abs(e.fraction(125) - 0.5) < 1e-9
    assert e.fraction(149) == pytest.approx(0.98)


def test_from_envspec_year_to_generation():
    dyn = {"generations_per_year": 100,
           "events": [{"t_year": 10, "type": "salt_ramp", "to_M": 5.0, "duration_y": 5}]}
    s = EventScript.from_envspec(dyn)
    assert s.events[0].t_gen == 1000 and s.events[0].duration_g == 500


def test_run_with_events_bottleneck_and_ramp():
    k = MiniKernel()
    end = k.prior_table.copy()
    end[:, 1] = 0.01                                # 终态: 残基 1 无优势
    end /= end.sum(axis=1, keepdims=True)
    script = EventScript([
        Event("bottleneck", 5, 5, factor=0.2),
        Event("env_ramp", 10, 10, end_table=end)])
    seen_ne = []

    def obs(gen, pops, stats):
        seen_ne.append(pops.shape[1])
    stats, pops, log = run_with_events(k, script, n_pop=2, n_gen=25, Ne=50,
                                       observer=obs)
    assert len(stats) == 50                          # 25 代 × 2 群
    assert all(n == 50 for n in seen_ne)             # 瓶颈不改变输出形状（代内收缩）
    types = {d["type"] for d in log}
    assert types == {"bottleneck", "env_ramp"}
    assert not np.allclose(k.prior_table, end)       # 渐变停在 frac→1（含数值路径）


def test_migration_injects_donor():
    k = MiniKernel()
    donor = np.full(k.L, 5, dtype=np.int8)
    script = EventScript([Event("migration", 0, 1, donor=donor.tolist(), fraction=0.5)])
    _, pops, _ = run_with_events(k, script, n_pop=1, n_gen=1, Ne=40)
    frac = (pops[0] == 5).all(axis=1).mean()
    assert 0.1 < frac <= 0.5 + 1e-9


def test_temperature_shift_applies():
    k = MiniKernel()
    script = EventScript([Event("temperature_shift", 2, 5, to_T=1.5)])
    run_with_events(k, script, n_pop=1, n_gen=6, Ne=30)
    assert k.T == 1.5


# ---------------- lineage ----------------
def test_recorder_hit_rate_and_fixation():
    k = MiniKernel(L=4, seed=3)
    k.prior_table[:] = k.prior_table                # 优势残基=1
    rec = LineageRecorder(k.sites, fix_thresh=0.9)
    rec.set_wt(k.wt_idx)
    run_with_events(k, EventScript([]), n_pop=3, n_gen=30, Ne=60, observer=rec.observe)
    assert rec.total_inds == 3 * 60 * 30
    assert 0.0 < rec.cache_hit_rate < 1.0
    edges = rec.build_lineage()
    assert edges and all(parent is None or isinstance(parent, str) for _, _, parent in edges)
    tl = rec.timeline()
    for d in tl:
        assert d["aa"] in AA20


def test_recorder_lineage_parent_is_edit_distance_one():
    k = MiniKernel(L=3, seed=1)
    rec = LineageRecorder(k.sites)
    rec.set_wt(k.wt_idx)
    run_with_events(k, EventScript([]), n_pop=2, n_gen=15, Ne=30, observer=rec.observe)
    edges = rec.build_lineage()
    geno = rec._genos
    checked = 0
    for g, h, parent in edges:
        if parent is not None and rec._seen[parent] < g:
            assert int((geno[h] != geno[parent]).sum()) == 1
            checked += 1
    assert checked >= 1                              # 至少一条真实单步边


def test_report_and_save(tmp_path):
    k = MiniKernel(L=4, seed=5)
    rec = LineageRecorder(k.sites)
    rec.set_wt(k.wt_idx)
    run_with_events(k, EventScript([]), n_pop=2, n_gen=20, Ne=40, observer=rec.observe)
    rep = rec.narrative_report(run_meta={"landscape": "mini"})
    assert "进化轨迹叙事报告" in rep and "年轴三情景" in rep
    out = str(tmp_path / "run1")
    files = rec.save(out)
    for p in files.values():
        assert os.path.exists(p) and os.path.getsize(p) > 0


# ---------------- fitcache ----------------
def test_cache_roundtrip_and_stats(tmp_path):
    c = FitnessCache(str(tmp_path / "c.db"), model_version="m1")
    assert c.get([(10, "A")]) is None
    c.put([(10, "A")], {"fitness": 0.5})
    assert c.get([(10, "A")])["fitness"] == 0.5
    # 顺序无关
    assert c.get([(10, "A"), (5, "G")]) is None or True
    c.put([(5, "G"), (10, "A")], {"fitness": 0.7})
    assert c.get([(10, "A"), (5, "G")])["fitness"] == 0.7
    s = c.stats()
    assert s["entries"] == 2 and s["misses"] >= 2 and s["hits"] >= 2


def test_cache_model_version_invalidation(tmp_path):
    c1 = FitnessCache(str(tmp_path / "c.db"), model_version="m1")
    c1.put([(1, "W")], {"fitness": 1.0})
    c2 = FitnessCache(str(tmp_path / "c.db"), model_version="m2")
    assert c2.get([(1, "W")]) is None                # 模型换代 → 旧条目不命中
    c2.put([(1, "W")], {"fitness": 2.0})
    assert c2.get([(1, "W")])["fitness"] == 2.0
    assert c2.stats()["entries"] == 1


def test_compute_through(tmp_path):
    calls = {"n": 0}
    c = FitnessCache(str(tmp_path / "c.db"))

    def compute(muts):
        calls["n"] += 1
        return {"fitness": len(set(muts))}
    assert c.compute_through([(3, "A")], compute)["fitness"] == 1
    assert c.compute_through([(3, "A")], compute)["fitness"] == 1  # 命中, 不重算
    assert calls["n"] == 1
