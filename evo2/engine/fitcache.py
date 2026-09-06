# -*- coding: utf-8 -*-
"""适应度缓存（v1 方案 §3.2 关键决策 2 —— "万年模拟的可行性核心"）。

用途: 昂贵层（PLM/结构 Ensemble）对基因型的评估结果持久化, 运行期命中即查表,
未命中才调模型。sqlite 单文件、无服务依赖; key = 基因型突变串的规范哈希,
value = JSON（适应度与任意元数据）, 同时记录 model_version——模型换代自动失效。

与 lineage.LineageRecorder.cache_hit_rate 的分工: 后者是"命中率的理论口径"
（重复个体占比, 无需真正调模型）; 本模块是昂贵层调用路径上的真实缓存。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = ["FitnessCache"]


def _norm_key(muts: Iterable[Tuple[int, str]]) -> str:
    s = ";".join(f"{int(p)}{a}" for p, a in sorted(set((int(p), a) for p, a in muts)))
    return hashlib.sha1(s.encode()).hexdigest()[:20]


class FitnessCache:
    def __init__(self, path: str, model_version: str = "v0"):
        self.path = path
        self.model_version = model_version
        self.hits = 0
        self.misses = 0
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " k TEXT, model TEXT, v TEXT, PRIMARY KEY (k, model))")
        self.db.commit()

    # ------------------------------------------------------------------
    def get(self, muts: Iterable[Tuple[int, str]]) -> Optional[dict]:
        k = _norm_key(muts)
        row = self.db.execute(
            "SELECT v FROM cache WHERE k=? AND model=?",
            (k, self.model_version)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, muts: Iterable[Tuple[int, str]], value: dict) -> None:
        k = _norm_key(muts)
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                        (k, self.model_version, json.dumps(value)))
        self.db.commit()

    # ------------------------------------------------------------------
    def get_many(self, keys: List[Iterable[Tuple[int, str]]]
                 ) -> List[Optional[dict]]:
        return [self.get(m) for m in keys]

    def compute_through(self, muts: Iterable[Tuple[int, str]], compute) -> dict:
        """缓存穿透式调用: 命中返回缓存, 未命中调 compute(muts) 并回填。"""
        got = self.get(muts)
        if got is not None:
            return got
        val = compute(muts)
        self.put(muts, val)
        return val

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def stats(self) -> Dict[str, float]:
        return dict(hits=self.hits, misses=self.misses,
                    hit_rate=round(self.hit_rate, 4),
                    entries=self.db.execute(
                        "SELECT COUNT(*) FROM cache WHERE model=?",
                        (self.model_version,)).fetchone()[0])

    def close(self):
        self.db.close()
