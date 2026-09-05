# -*- coding: utf-8 -*-
"""EnvSpec schema 校验（v1 方案 §3.1 五层结构）。

校验从宽: 只锁"机器可执行适应度定义"所必需的结构（五层齐、类型对、数值合法）,
不锁具体字段名——词表（vocabulary.py）负责字段与机制的绑定, LLM 编译器负责产出。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

__all__ = ["validate_envspec", "load_envspec", "EnvSpecError"]

SECTIONS = ("protein", "environment", "fitness", "dynamics", "safety")


class EnvSpecError(ValueError):
    """EnvSpec 结构非法（附全部错误列表）。"""


def validate_envspec(spec: Any) -> List[str]:
    """返回错误列表（空 = 合法）。刻意返回 list 而非抛异常, 便于编译器逐条回喂 LLM。"""
    errors: List[str] = []
    if not isinstance(spec, dict):
        return ["EnvSpec 必须是 JSON 对象"]

    for sec in SECTIONS:
        if sec not in spec:
            errors.append(f"缺少必需节: {sec}")
    if errors:
        return errors  # 缺节时不再深查, 避免雪崩

    # ---- protein ----
    p = spec["protein"]
    if not isinstance(p, dict) or not p.get("name"):
        errors.append("protein.name 必填（种子蛋白名称）")
    if isinstance(p, dict) and not (p.get("source_pdb") or p.get("wt_fasta")):
        errors.append("protein 需要 source_pdb 或 wt_fasta 之一（种子输入）")

    # ---- environment ----
    env = spec["environment"]
    if not isinstance(env, dict) or not str(env.get("description", "")).strip():
        errors.append("environment.description 必填（自然语言环境原句, 供报告溯源）")

    # ---- fitness ----
    fit = spec["fitness"]
    if not isinstance(fit, dict):
        errors.append("fitness 必须是对象")
    else:
        comps = fit.get("components")
        if not isinstance(comps, dict) or not comps:
            errors.append("fitness.components 必须是非空的多分量加权和表")
        else:
            total = 0.0
            for k, v in comps.items():
                if not isinstance(v, dict) or not isinstance(v.get("weight"), (int, float)):
                    errors.append(f"fitness.components.{k}.weight 必须是数值")
                else:
                    total += float(v["weight"])
            if errors == [] and not (0.5 <= total <= 6.0):
                # 不强制归一（PprI 实例权重和=3.4）, 只拦明显失真
                errors.append(f"fitness.components 权重和 {total:.2f} 异常（应在 0.5–6.0）")

    # ---- dynamics ----
    dyn = spec["dynamics"]
    if not isinstance(dyn, dict):
        errors.append("dynamics 必须是对象")
    else:
        if not str(dyn.get("model", "")).strip():
            errors.append("dynamics.model 必填（如 Wright-Fisher）")
        ne = dyn.get("Ne")
        if not isinstance(ne, int) or ne <= 0:
            errors.append("dynamics.Ne 必须是正整数")

    # ---- safety ----
    saf = spec["safety"]
    if not isinstance(saf, dict) or not str(saf.get("target_class", "")).strip():
        errors.append("safety.target_class 必填（生物安全审核字段, v1 §9）")
    if isinstance(saf, dict) and not isinstance(saf.get("allowed"), bool):
        errors.append("safety.allowed 必须是布尔值（拒绝逻辑独立于 LLM, v1 §3.6）")

    return errors


def load_envspec(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    errs = validate_envspec(spec)
    if errs:
        raise EnvSpecError(f"{path} 不是合法 EnvSpec: {'; '.join(errs)}")
    return spec
