# -*- coding: utf-8 -*-
"""pytest 路径注入：repo root(engine) + esm3(代理) + benchmarks(b7)。"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # evo2/
for p in (REPO,
          os.path.join(REPO, "esm3"),
          os.path.join(REPO, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)
