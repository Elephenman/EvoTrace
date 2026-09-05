# -*- coding: utf-8 -*-
"""预设环境库 —— 用户自选模板 + 自输条件（2026-09-05 拍板, 取代自然语言编译器）。

流程: 选预设 → apply_overrides 逐项改条件 → build_envspec 填入种子蛋白 → schema 校验。
设计约束:
  1. 预设只含 environment/fitness/dynamics/safety 四节——种子蛋白(protein 节)必须
     由用户指定, 预设不越权;
  2. override 只允许改词表(vocabulary.VOCAB)里有的键, 且无机制参数
     (NONE_TERMS, 如磁场)一律拒绝并提示——把 v1 的幻觉防线条款变成硬校验;
  3. 每个预设都应通过 validate_envspec（拼上任意合法 protein 节后）。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from .schema import validate_envspec
from .vocabulary import NONE_TERMS, VOCAB

__all__ = ["PRESETS", "preset_names", "apply_overrides", "build_envspec", "PresetError"]


class PresetError(ValueError):
    """预设名不存在 / override 非法。"""


# ---------------------------------------------------------------- 预设模板
# 各节结构以 PprI 真实战役规范 ppri_evo/config/envspec.json 为蓝本泛化。

_HIGH_SALT = "high_salt_alkaline"

PRESETS: Dict[str, Dict[str, Any]] = {
    # v1 方案用户故事原句场景: 嗜热碱性蛋白酶 + 高盐碱地
    _HIGH_SALT: {
        "description": "高盐碱地: 5 M NaCl、pH 9.5、60 ℃、50 mM Mg2+（v1 方案用户故事场景）",
        "environment": {
            "description": "5 M NaCl, pH 9.5, 60 C, 50 mM MgCl2",
            "temperature_C": 60.0,
            "ph": 9.5,
            "salts": [{"ion": "NaCl", "M": 5.0}, {"ion": "Mg2+", "mM": 50.0}],
            "halotolerance_weight_hint": "surface_acidic_residues",  # 嗜盐机制先验标签
        },
        "fitness": {
            "type": "multi_component_weighted_with_gates",
            "components": {
                "stability":   {"definition": "热稳定性 ΔTm/ΔΔG (RaSP/ThermoMPNN 口径)",
                                 "tier": "mid", "weight": 0.30},
                "activity":    {"definition": "碱性条件催化效率代理 (家族 PLM)", 
                                 "tier": "mid", "weight": 0.35},
                "halotolerance": {"definition": "表面酸性残基比例 + 盐桥密度先验",
                                   "tier": "cheap", "weight": 0.20},
                "solubility":  {"definition": "CamSol 类序列聚集先验",
                                 "tier": "cheap", "weight": 0.15},
            },
            "hard_gates": [{"property": "tm_C", "op": ">=", "value": 62}],
        },
        "dynamics": {
            "model": "Wright-Fisher",
            "Ne": 100000,
            "mutations_per_genome_per_gen": {"dist": "poisson", "lambda": 0.6},
            "generations_per_year": 300,
            "years": 10000,
            "events": [{"t_year": 4000, "type": "salt_ramp", "to_M": 5.0, "duration_y": 500}],
            "proposal_mode": "faithful",
        },
        "safety": {"target_class": "industrial_protease", "allowed": True},
    },

    # 常温中性默认环境（工厂底盘菌表达场景）
    "mesophilic_neutral": {
        "description": "常温中性水相: 25 ℃、pH 7.0、无附加胁迫（基线/对照场景）",
        "environment": {
            "description": "water, 25 C, pH 7.0",
            "temperature_C": 25.0,
            "ph": 7.0,
            "salts": [],
        },
        "fitness": {
            "type": "multi_component_weighted_with_gates",
            "components": {
                "stability":  {"definition": "ΔΔG vs WT", "tier": "mid", "weight": 0.35},
                "activity":   {"definition": "家族 PLM 活性代理", "tier": "mid", "weight": 0.35},
                "solubility": {"definition": "聚集先验", "tier": "cheap", "weight": 0.15},
                "expressibility": {"definition": "密码子适配 + PLM 表达头", "tier": "cheap",
                                    "weight": 0.15},
            },
        },
        "dynamics": {
            "model": "Wright-Fisher",
            "Ne": 100000,
            "mutations_per_genome_per_gen": {"dist": "poisson", "lambda": 0.6},
            "generations_per_year": 300,
            "years": 1000,
            "proposal_mode": "faithful",
        },
        "safety": {"target_class": "industrial_enzyme", "allowed": True},
    },

    # 氧化胁迫（洗涤剂/工业氧化环境）
    "oxidative_stress": {
        "description": "氧化胁迫水相: 25 ℃、pH 7.5、H2O2 胁迫（Met/Cys 暴露敏感场景）",
        "environment": {
            "description": "water, 25 C, pH 7.5, oxidative stress (peroxide)",
            "temperature_C": 25.0,
            "ph": 7.5,
            "oxidative_stress": True,
        },
        "fitness": {
            "type": "multi_component_weighted_with_gates",
            "components": {
                "oxidation_resistance": {"definition": "Met/Cys 暴露面积先验",
                                          "tier": "cheap", "weight": 0.40},
                "stability":  {"definition": "ΔΔG vs WT", "tier": "mid", "weight": 0.25},
                "activity":   {"definition": "家族 PLM 活性代理", "tier": "mid", "weight": 0.35},
            },
        },
        "dynamics": {
            "model": "Wright-Fisher",
            "Ne": 100000,
            "mutations_per_genome_per_gen": {"dist": "poisson", "lambda": 0.6},
            "generations_per_year": 300,
            "years": 2000,
            "proposal_mode": "faithful",
        },
        "safety": {"target_class": "industrial_enzyme", "allowed": True},
    },

    # DNA 损伤应答环境（PprI 战役定型条件的蛋白无关泛化版）
    "nucleic_acid_discrimination": {
        "description": "核酸底物判别环境: 催化金属 + 靶/非靶核酸条件差分（PprI 战役泛化）",
        "environment": {
            "description": "catalytic metal cofactor present; target vs non-target "
                           "nucleic-acid substrates as fitness conditions",
            "ions": [{"ion": "Mn2+", "mM": 2.0}],
            "nucleic_acid": {
                "target": "<用户指定靶序列>",
                "nontarget_panel": ["<同组成打乱对照>", "<单碱基探针>"],
                "sep_definition": "act(target) - act(nontarget)",
            },
        },
        "fitness": {
            "type": "multi_component_weighted_with_gates",
            "components": {
                "target_activation":  {"definition": "靶底物结合/激活态概率 (结构 Ensemble 口径)",
                                        "tier": "expensive", "weight": 1.0},
                "nontarget_suppression": {"definition": "1 - 非靶底物同口径概率",
                                           "tier": "expensive", "weight": 0.9},
                "sequence_naturalness": {"definition": "PLM ΔLL vs WT", "tier": "mid",
                                          "weight": 0.5},
                "fold_stability": {"definition": "ΔΔG vs WT (硬门槛可加)", "tier": "mid",
                                    "weight": 0.4},
            },
        },
        "dynamics": {
            "model": "Wright-Fisher",
            "Ne": 500,
            "mutations_per_genome_per_gen": {"dist": "poisson", "lambda": 0.6},
            "generations_wave0": 15,
            "selection": "softmax(fitness / T), T=0.6",
            "proposal_mode": "faithful",
            "expensive_tier": {"every_k_generations": 5, "top_k_per_pop": 6, "samples": 20},
        },
        "safety": {"target_class": "dna_repair_protein_engineering_in_vitro", "allowed": True},
    },

    # 高温胁迫
    "thermal_stress": {
        "description": "高温胁迫: 80 ℃、pH 7.0（嗜热改造场景）",
        "environment": {
            "description": "water, 80 C, pH 7.0",
            "temperature_C": 80.0,
            "ph": 7.0,
        },
        "fitness": {
            "type": "multi_component_weighted_with_gates",
            "components": {
                "stability": {"definition": "ΔTm/ΔΔG, 硬门槛 Tm>=75", "tier": "mid",
                               "weight": 0.50},
                "activity":  {"definition": "高温下活性保持代理", "tier": "mid", "weight": 0.35},
                "solubility": {"definition": "聚集先验", "tier": "cheap", "weight": 0.15},
            },
            "hard_gates": [{"property": "tm_C", "op": ">=", "value": 75}],
        },
        "dynamics": {
            "model": "Wright-Fisher",
            "Ne": 100000,
            "mutations_per_genome_per_gen": {"dist": "poisson", "lambda": 0.6},
            "generations_per_year": 300,
            "years": 5000,
            "proposal_mode": "faithful",
        },
        "safety": {"target_class": "industrial_enzyme", "allowed": True},
    },
}


def preset_names() -> list:
    return list(PRESETS)


# ---------------------------------------------------------------- 构造 API

def apply_overrides(preset_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """取预设模板并应用用户修改的条件参数, 返回四节 dict（未含 protein 节）。

    override 键规则:
      - 顶层允许改 environment/fitness/dynamics/safety 四节（值整节替换, 供进阶用户）;
      - environment 内的键必须在词表 VOCAB 中且 mechanism != none（无机制参数拒绝）;
      - 其余位置不允许点路径式深改（避免绕过词表防线）。
    """
    if preset_name not in PRESETS:
        raise PresetError(f"预设 {preset_name!r} 不存在, 可选: {preset_names()}")
    spec = copy.deepcopy(PRESETS[preset_name])
    env = spec["environment"]
    for k, v in (overrides or {}).items():
        if k in ("environment", "fitness", "dynamics", "safety"):
            spec[k] = copy.deepcopy(v)
        elif k not in VOCAB:
            raise PresetError(f"参数 {k!r} 不在环境词表中, 可用: {sorted(VOCAB)}")
        elif k in NONE_TERMS:
            raise PresetError(
                f"参数 {k!r} 无适应度机制绑定（词表标注 none）, 已拒绝——"
                f"如确需该条件请先在 vocabulary.py 补机制依据")
        elif k in env:
            env[k] = copy.deepcopy(v)
        else:
            raise PresetError(
                f"参数 {k!r} 在词表中但预设 {preset_name!r} 未启用该条件, "
                f"如需新增请改整个 environment 节")
    return spec


def build_envspec(preset_name: str, protein: Dict[str, Any],
                  overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """预设 + 用户种子蛋白 + 条件修改 → 校验通过的完整 EnvSpec。

    protein 至少含 name 与 (source_pdb 或 wt_fasta)。校验失败抛 EnvSpecError。
    """
    spec = apply_overrides(preset_name, overrides)
    spec["protein"] = copy.deepcopy(protein or {})
    errs = validate_envspec(spec)
    if errs:
        from .schema import EnvSpecError
        raise EnvSpecError("EnvSpec 校验失败: " + "; ".join(errs))
    return spec
