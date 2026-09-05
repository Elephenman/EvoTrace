# -*- coding: utf-8 -*-
"""环境词表 v0 —— 20 条（v1 方案 §3.1 / 附录第 2 条）。

每条环境参数绑定到**有据可依的适应度机制**; 无机制对应的参数（如"磁场"）必须显式
标注 mechanism="none", 由编译器向用户披露并忽略——这是防止"语言模型幻觉直接变成
适应度"的关键条款。

词表字段:
  key       EnvSpec 内的规范键名
  mechanism 绑定的适应度机制（M1/M2 层的消费方）; "none" = 无机制, 编译期忽略
  tier      消费层: cheap(本地/先验) | mid(PLM) | expensive(结构/ Ensemble)
"""
from __future__ import annotations

VOCAB = {
    # ---- 物理化学（M1 环境编译层）----
    "temperature_C":  {"mechanism": "stability_component(ΔTm gate + 热失活项)", "tier": "cheap"},
    "ph":             {"mechanism": "activity_component(最适 pH 偏移项)", "tier": "cheap"},
    "salts":          {"mechanism": "halotolerance_component(表面酸性残基/盐桥先验 + 离子结合项)", "tier": "cheap"},
    "ions":           {"mechanism": "cofactor_component(催化金属配位, 如 PprI Mn2+)", "tier": "expensive"},
    "oxidative_stress": {"mechanism": "oxidation_component(Met/Cys 暴露面积先验)", "tier": "cheap"},
    "pressure_atm":   {"mechanism": "packing_component(空腔体积/压缩先验)", "tier": "cheap"},
    "solvent":        {"mechanism": "solvation_component(有机溶剂暴露耐受先验)", "tier": "cheap"},
    "ligand":         {"mechanism": "binding_component(结构共折叠亲和, Boltz-2 口径)", "tier": "expensive"},
    "nucleic_acid":   {"mechanism": "dna_condition_component(靶/非靶条件差分, PprI 战役定型)", "tier": "expensive"},
    # ---- 进化动力学（M3 层）----
    "host":           {"mechanism": "mutation_spectrum(物种特异替换谱) + 世代时间换算", "tier": "cheap"},
    "Ne":             {"mechanism": "drift_strength(Wright-Fisher 采样)", "tier": "cheap"},
    "generations_per_year": {"mechanism": "time_axis(三情景年轴换算, v1 §3.5)", "tier": "cheap"},
    "years":          {"mechanism": "time_axis(总代数 = years × generations_per_year)", "tier": "cheap"},
    "mutation_rate":  {"mechanism": "mutation_operator(每基因组每代突变数分布)", "tier": "cheap"},
    "event_salt_ramp": {"mechanism": "event_script(环境渐变: 盐浓度随时间线性上升)", "tier": "cheap"},
    "event_bottleneck": {"mechanism": "event_script(种群瓶颈: Ne 骤降)", "tier": "cheap"},
    "event_migration": {"mechanism": "event_script(基因流/HGT: 引入外源同源)", "tier": "cheap"},
    "event_temperature_shift": {"mechanism": "event_script(温度台阶/渐变)", "tier": "cheap"},
    # ---- 安全（独立于 LLM 的规则引擎兜底, v1 §3.6/§9）----
    "target_class":   {"mechanism": "safety_whitelist(允许类别白名单 + 毒素/毒力/抗性拒绝)", "tier": "cheap"},
    # ---- 显式无机制示例（编译器必须标注 ignored 并向用户披露）----
    "magnetic_field": {"mechanism": "none", "tier": "cheap"},
}

CHEAP_TERMS = [k for k, v in VOCAB.items() if v["tier"] == "cheap"]
EXPENSIVE_TERMS = [k for k, v in VOCAB.items() if v["tier"] == "expensive"]
NONE_TERMS = [k for k, v in VOCAB.items() if v["mechanism"] == "none"]


def mechanism_of(key: str) -> str:
    return VOCAB.get(key, {}).get("mechanism", "unknown")


def prompt_block() -> str:
    """给 LLM 编译器的词表提示块（compiler.build_prompt 拼接用）。"""
    lines = ["可用环境参数（key → 机制绑定 @ 层）:"]
    for k, v in VOCAB.items():
        mark = "  [无机制→必须标注 ignored]" if v["mechanism"] == "none" else ""
        lines.append(f"  {k}: {v['mechanism']} @ {v['tier']}{mark}")
    return "\n".join(lines)
