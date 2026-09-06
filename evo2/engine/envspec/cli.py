#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EnvSpec 交互壳 —— 选预设 → 自输条件 → 确认落盘（路线图 R1 收尾项）。

运行: python evo2/engine/envspec/cli.py [输出路径]
流程: 列预设 → 选号 → 显示参数 → 逐项修改（回车跳过）→ 输入种子蛋白 → 校验 → 确认。
非交互场景用库 API: build_envspec(preset, protein, overrides)（见 presets.py）。
"""
import json
import sys

from .presets import PRESETS, PresetError, apply_overrides, preset_names
from .schema import EnvSpecError, validate_envspec
from .vocabulary import VOCAB


def _ask(prompt, default=None, cast=str):
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  ! 无法解析 {raw!r}, 保留默认 {default!r}")
        return default


def _show_env(env):
    for k, v in env.items():
        print(f"  {k:24s} = {v}")


def main(out_path: str = "envspec_user.json"):
    names = preset_names()
    print("== EvoTrace 预设环境 ==")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}: {PRESETS[n]['description']}")
    idx = _ask(f"选择预设 [0-{len(names)-1}] (默认 0): ", 0, int)
    name = names[idx]

    spec = apply_overrides(name, {})
    print(f"\n== 当前条件（{name}）==")
    _show_env(spec["environment"])

    print("\n可直接回车保留各项; 输入新值即修改（词表键: temperature_C/ph/salts 等）")
    overrides = {}
    while True:
        key = input("要修改的环境参数 (回车=不改了): ").strip()
        if not key:
            break
        try:
            raw = input(f"  {key} 的新值 (JSON, 如 65.0): ")
            val = json.loads(raw) if raw.strip() else None
            overrides[key] = val
            spec = apply_overrides(name, overrides)   # 每次从预设重放, 保持词表防线
            print(f"  ✓ {key} = {spec['environment'].get(key)}")
        except (PresetError, json.JSONDecodeError) as e:
            print(f"  ! {e}")
            overrides.pop(key, None)

    protein = {
        "name": _ask("种子蛋白名称: ", "user_protein"),
        "wt_fasta": _ask("WT FASTA 路径 (或 source_pdb): ", "inputs/wt.fasta"),
    }
    spec["protein"] = protein
    errs = validate_envspec(spec)
    if errs:
        print("\n校验失败:", "; ".join(errs))
        return 1
    print(f"\n校验通过。确认写出到 {out_path}? [Y/n]")
    if input().strip().lower() in ("n", "no"):
        print("已取消（v1 §3.1 确认条款）")
        return 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)
    print(f"已写出: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "envspec_user.json"))
