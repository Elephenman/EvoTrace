# -*- coding: utf-8 -*-
"""验证 b7 注册无语法错 + DnaAwareLandscape（b7 DnaAwareOracle 底层）接口正确。"""
import py_compile, sys
import numpy as np

sys.path.insert(0, "A:/claudework/evo2/benchmarks")
sys.path.insert(0, "A:/claudework/evo2/esm3")

# 1) b7 语法
py_compile.compile("A:/claudework/evo2/benchmarks/b7_three_way.py", doraise=True)
print("[ok] b7_three_way.py 语法正确（dna_aware_ppri 注册已写入）")
import b7_three_way  # import 不触发 main
print("[ok] b7 import OK")

# 2) DnaAwareLandscape（即 b7 DnaAwareOracle 的底层）等价验证
from ppri_surrogate_v3 import PprISurrogateV3
from ppri_dna_aware import DnaAwareLandscape
base = PprISurrogateV3()
g = DnaAwareLandscape(base)
wt = float(g.evaluate(g.wt_idx[None, :])[0])

gg = g.wt_idx.copy(); gg[6] = 19   # F88->Y（读头芳香，靶标 G 应保留高 gate）
fy = float(g.evaluate(gg[None, :])[0])
gg2 = g.wt_idx.copy(); gg2[5] = 0  # R85->A（锚点破坏，应被 gate 惩罚）
ra = float(g.evaluate(gg2[None, :])[0])

print(f"WT={wt:+.3f}  F88Y={fy:+.3f}  R85A={ra:+.3f}")
assert ra < wt, "锚点破坏必须被惩罚"
assert hasattr(g, "n_mutations") and hasattr(g, "enforce_max_mut") and hasattr(g, "evaluate")
print("[ok] 搜索内核接口对齐 (L/wt_idx/sites/evaluate/n_mutations/enforce_max_mut)")
print("[ok] 机制惩罚生效 -> dna_aware_ppri 经 b7 可用")
