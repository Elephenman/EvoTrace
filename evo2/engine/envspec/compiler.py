# -*- coding: utf-8 -*-
"""EnvSpecCompiler —— 自然语言 → EnvSpec 编译器 v0（v1 方案 §3.1, 验收线 1）。

隔离原则: LLM 只做"语言 → 结构化参数", 数值由词表机制给出; 编译器自身:
  1. build_prompt(): 把 schema 约定 + 词表机制绑定拼成函数调用式提示;
  2. compile(): 调用注入的 llm callable（任意后端: API/本地/测试桩）, 取回 JSON;
  3. validate: schema 校验失败 → 错误列表回喂 llm 最多 max_retries 次;
  4. confirm: 可选确认回调（交互式界面挂点）, 用户确认前不返回终稿。
无机制参数（vocabulary.NONE_TERMS）编译期自动剥离并生成 disclosures 报告。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schema import validate_envspec
from .vocabulary import NONE_TERMS, prompt_block

__all__ = ["EnvSpecCompiler", "CompileResult"]

SYSTEM_PROMPT = (
    "你是环境规范编译器。把用户的自然语言环境描述编译成严格符合 schema 的 EnvSpec JSON。\n"
    "硬规则: 1) 只产结构化参数与机制标签, 不做任何数值预测;\n"
    "2) 只使用词表中列出的环境参数键;\n"
    "3) 词表中 mechanism=none 的参数一律不写入 EnvSpec;\n"
    "4) 无法从原句推断的数值字段必须留空并加入 open_questions, 不得编造。\n"
    "输出: 单个 JSON 对象, 不要多余文本。"
)


class CompileResult:
    def __init__(self, spec: Dict[str, Any], disclosures: List[str],
                 open_questions: List[str], retries: int):
        self.spec = spec
        self.disclosures = disclosures      # 被剥离的无机制参数, 向用户披露
        self.open_questions = open_questions  # LLM 声明无法推断的字段
        self.retries = retries

    def __repr__(self) -> str:
        return (f"<CompileResult spec={sorted(self.spec)} "
                f"disclosures={self.disclosures} retries={self.retries}>")


class EnvSpecCompiler:
    def __init__(self, llm: Callable[[str], str], max_retries: int = 2):
        """llm: prompt(str) -> 模型回复(str, 期望为纯 JSON)。
        任意后端以 callable 注入, 测试用桩、生产接 API, 编译器不感知。"""
        self.llm = llm
        self.max_retries = max_retries

    def build_prompt(self, text: str) -> str:
        return (f"{SYSTEM_PROMPT}\n\n{prompt_block()}\n\n"
                f"schema 必需五节: protein/environment/fitness/dynamics/safety\n"
                f"fitness.components 每项含 weight(数值) 与 tier;\n"
                f"dynamics 需 model 与 Ne(正整数); safety 需 target_class 与 allowed(布尔)。\n\n"
                f"用户环境描述: {text}")

    @staticmethod
    def _strip_disclosures(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """剥离无机制参数（对 environment 顶层键做过滤）。"""
        disclosures = []
        if isinstance(spec.get("environment"), dict):
            for k in list(spec["environment"]):
                if k in NONE_TERMS:
                    disclosures.append(k)
                    del spec["environment"][k]
        return spec, disclosures

    def compile(self, text: str,
                confirm: Optional[Callable[[Dict[str, Any]], bool]] = None) -> CompileResult:
        prompt = self.build_prompt(text)
        retries = 0
        errors: List[str] = []
        while True:
            raw = self.llm(prompt if not errors else
                           prompt + f"\n\n上次输出未通过校验: {'; '.join(errors)}\n请修正后重新输出完整 JSON。")
            try:
                spec = json.loads(raw)
                if not isinstance(spec, dict):
                    raise ValueError("顶层不是 JSON 对象")
            except (json.JSONDecodeError, ValueError) as e:
                errors = [f"JSON 解析失败: {e}"]
                retries += 1
                if retries > self.max_retries:
                    raise ValueError(f"编译器连续 {self.max_retries + 1} 次未产出合法 JSON: {errors}")
                continue
            spec, disclosures = self._strip_disclosures(spec)
            errors = validate_envspec(spec)
            if not errors:
                break
            retries += 1
            if retries > self.max_retries:
                raise ValueError(f"EnvSpec 校验连续失败: {'; '.join(errors)}")

        open_q = spec.pop("open_questions", []) or []
        if confirm is not None and not confirm(spec):
            raise ValueError("用户未确认该 EnvSpec（v1 §3.1 交互确认条款）")
        return CompileResult(spec, disclosures, list(open_q), retries)
