# -*- coding: utf-8 -*-
"""预设环境库测试 —— 验收线 1 修订版（预设选项 + 用户自输条件, 2026-09-05 拍板）。

出口标准（路线图 R1 修订）: 每个预设拼上合法 protein 节后通过 schema;
override 只认词表参数; 无机制参数拒绝; 非法 protein 被校验拦下。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envspec import (PRESETS, PresetError, apply_overrides, build_envspec,
                     preset_names)
from envspec.schema import EnvSpecError, validate_envspec
from envspec.vocabulary import VOCAB

PROTEIN = {"name": "alkaline_protease", "wt_fasta": "inputs/wt.fasta"}


def test_five_presets_all_valid_with_any_protein():
    assert len(PRESETS) == 5
    for name in preset_names():
        spec = build_envspec(name, PROTEIN)
        assert validate_envspec(spec) == []
        assert spec["protein"]["name"] == "alkaline_protease"
        assert spec["safety"]["allowed"] is True


def test_preset_templates_do_not_carry_protein():
    for name in preset_names():
        assert "protein" not in PRESETS[name], "预设不得预设种子蛋白"


def test_high_salt_preset_matches_v1_user_story():
    spec = build_envspec("high_salt_alkaline", PROTEIN)
    env = spec["environment"]
    assert env["salts"][0] == {"ion": "NaCl", "M": 5.0}
    assert env["ph"] == 9.5 and env["temperature_C"] == 60.0
    assert spec["dynamics"]["years"] == 10000
    assert any(e["type"] == "salt_ramp" for e in spec["dynamics"]["events"])


def test_override_env_param():
    spec = build_envspec("mesophilic_neutral", PROTEIN,
                         {"temperature_C": 37.0, "ph": 8.0})
    assert spec["environment"]["temperature_C"] == 37.0
    assert spec["environment"]["ph"] == 8.0
    assert validate_envspec(spec) == []


def test_override_rejects_unknown_and_none_mechanism_params():
    with pytest.raises(PresetError, match="不在环境词表"):
        apply_overrides("mesophilic_neutral", {"gravity": 0.0})
    with pytest.raises(PresetError, match="无适应度机制"):
        apply_overrides("mesophilic_neutral", {"magnetic_field": True})


def test_override_rejects_bad_preset_name():
    with pytest.raises(PresetError, match="不存在"):
        apply_overrides("no_such_env", {})


def test_build_rejects_invalid_protein():
    with pytest.raises(EnvSpecError):
        build_envspec("mesophilic_neutral", {"name": "x"})  # 缺种子输入
    with pytest.raises(EnvSpecError):
        build_envspec("mesophilic_neutral", None)


def test_whole_section_override_for_power_users():
    dyn = dict(PRESETS["mesophilic_neutral"]["dynamics"])
    dyn["Ne"] = 500
    spec = build_envspec("mesophilic_neutral", PROTEIN, {"dynamics": dyn})
    assert spec["dynamics"]["Ne"] == 500
    assert validate_envspec(spec) == []


def test_all_env_keys_are_in_vocabulary():
    """防线一致性: 任何预设写进 environment 的键都必须能在词表里说清机制。"""
    for name in preset_names():
        for k in PRESETS[name]["environment"]:
            if k == "description":
                continue
            assert k in VOCAB or k.endswith("_hint") or k == "nucleic_acid", \
                f"{name}.{k} 不在词表（若为新参数请先补 vocabulary 机制绑定）"
