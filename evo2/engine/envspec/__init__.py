# -*- coding: utf-8 -*-
"""EnvSpec —— 自然语言环境编译器的 schema / 词表 / 编译器（v1 方案 §3.1, 验收线 1）。

隔离原则（v1 §3.1 硬约束）: LLM 只负责"语言 → 结构化参数 + 物理化学直觉标签",
**不做任何数值预测**——数值全部由词表绑定的适应度机制给出。
第一个校验实例: campaigns 时代的真实环境规范
`ppri_evo/config/envspec.json`（PprI DNA 损伤应答环境）。
"""
from .schema import EnvSpecError, load_envspec, validate_envspec
from .compiler import EnvSpecCompiler

__all__ = ["EnvSpecError", "load_envspec", "validate_envspec", "EnvSpecCompiler"]
