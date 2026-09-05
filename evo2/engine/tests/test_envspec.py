# -*- coding: utf-8 -*-
"""EnvSpec 编译器测试 —— 验收线 1（v1 方案 §3.1）。

关键验收: 真实战役环境规范 `ppri_evo/config/envspec.json` 通过 schema;
词表 20 条且含显式无机制条款; 编译器以桩 LLM 离线编译通过 + 无机制参数剥离披露 +
校验回喂重试 + 用户确认门。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envspec import EnvSpecCompiler, load_envspec, validate_envspec
from envspec.vocabulary import CHEAP_TERMS, NONE_TERMS, VOCAB, prompt_block

ENVSPEC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "ppri_evo", "config", "envspec.json"))


def test_real_ppri_envspec_passes_schema():
    spec = load_envspec(ENVSPEC)  # 不抛 = 合法
    assert spec["protein"]["name"].startswith("PprI")
    assert spec["dynamics"]["Ne"] == 500
    assert spec["safety"]["allowed"] is True


def test_broken_spec_lists_all_errors():
    errs = validate_envspec({"protein": {"name": "x"}, "environment": {}})
    assert any("缺少必需节" in e for e in errs)
    errs2 = validate_envspec({
        "protein": {}, "environment": {}, "fitness": {"components": {}},
        "dynamics": {"Ne": -1}, "safety": {}})
    joined = "; ".join(errs2)
    for frag in ("source_pdb", "description", "components", "dynamics.model",
                 "Ne", "target_class", "allowed"):
        assert frag in joined, f"缺少对 {frag} 的报错: {joined}"


def test_vocabulary_has_20_terms_and_none_clause():
    assert len(VOCAB) == 20
    assert "magnetic_field" in NONE_TERMS          # 幻觉防线条款
    assert all(v["mechanism"] != "unknown" for v in VOCAB.values())
    assert CHEAP_TERMS and len(CHEAP_TERMS) < 20   # 分层非空
    assert "mechanism=none" in prompt_block().replace(" ", "") or \
           "无机制" in prompt_block()


# ---- 编译器（桩 LLM, 离线）----

GOOD_SPEC = json.load(open(ENVSPEC, encoding="utf-8"))
GOOD_RAW = json.dumps(GOOD_SPEC)


def test_compile_with_stub_llm():
    c = EnvSpecCompiler(llm=lambda p: GOOD_RAW)
    r = c.compile("让 PprI 在 DNA 损伤样细胞质中区分靶/非靶 ssDNA")
    assert r.retries == 0 and r.disclosures == []
    assert r.spec["dynamics"]["Ne"] == 500


def test_compile_strips_none_mechanism_param():
    spec = json.loads(GOOD_RAW)
    spec["environment"]["magnetic_field"] = True   # 词表中的无机制参数
    raw = json.dumps(spec)
    c = EnvSpecCompiler(llm=lambda p: raw)
    r = c.compile("高盐环境外加磁场")
    assert r.disclosures == ["magnetic_field"]
    assert "magnetic_field" not in r.spec["environment"]


def test_compile_feeds_validation_errors_back():
    calls = {"n": 0}

    def llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"protein": {"name": "x"}})  # 缺四节 → 触发回喂
        return GOOD_RAW

    c = EnvSpecCompiler(llm=llm)
    r = c.compile("一句环境描述")
    assert r.retries == 1
    assert calls["n"] == 2


def test_compile_requires_user_confirmation():
    c = EnvSpecCompiler(llm=lambda p: GOOD_RAW)
    with pytest.raises(ValueError, match="未确认"):
        c.compile("描述", confirm=lambda spec: False)
    r = c.compile("描述", confirm=lambda spec: True)
    assert r.spec["safety"]["target_class"]


def test_compile_gives_up_after_max_retries():
    c = EnvSpecCompiler(llm=lambda p: "这不是JSON", max_retries=1)
    with pytest.raises(ValueError, match="合法 JSON"):
        c.compile("描述")
