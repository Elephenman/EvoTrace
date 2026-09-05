# -*- coding: utf-8 -*-
"""EnvSpec —— 环境规范模块（v1 方案 §3.1 结构, 验收线 1 修订版）。

流程（2026-09-05 用户拍板）: **不做自然语言编译器**。使用者自己输入条件——
从预设环境库（presets.py）选一个模板, 逐项修改参数, 指定种子蛋白, schema 校验
通过即得 EnvSpec。schema 与词表沿用 v1 的结构约定与"无机制参数必须披露"条款。
"""
from .schema import EnvSpecError, load_envspec, validate_envspec
from .presets import PRESETS, PresetError, apply_overrides, build_envspec, preset_names

__all__ = ["EnvSpecError", "load_envspec", "validate_envspec",
           "PRESETS", "PresetError", "preset_names", "apply_overrides", "build_envspec"]
