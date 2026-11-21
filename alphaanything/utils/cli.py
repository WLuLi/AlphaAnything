from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return result


def _parse_generic_overrides(values: Iterable[str]) -> dict[str, Any]:
    tokens = list(values)
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected CLI argument: {token}")
        expr = token[2:]
        if "=" in expr:
            key, raw_value = expr.split("=", 1)
        else:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ValueError(f"Override --{expr} requires a value")
            key, raw_value = expr, tokens[i + 1]
            i += 1
        overrides[key] = yaml.safe_load(raw_value)
        i += 1
    return overrides


def load_resolved_config(config_path: str | Path, overrides: dict[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    for key, value in overrides.items():
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return config


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def resolve_cli_config(args: argparse.Namespace, unknown: Iterable[str]) -> dict[str, Any]:
    overrides = _parse_generic_overrides(unknown)
    explicit = {
        "data.manifest_path": args.manifest,
        "model.pretrained_model_name_or_path": args.model,
        "checkpoint.output_dir": args.output_dir,
        "checkpoint.resume_from": args.resume_from,
        "training.max_train_steps": args.max_train_steps,
        "runtime.seed": args.seed,
    }
    for key, value in explicit.items():
        if value is not None:
            overrides[key] = value
    if args.local_files_only:
        overrides["model.local_files_only"] = True
    return load_resolved_config(args.config, overrides)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, dst)


def write_yaml(path: str | Path, value: Any) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    tmp.write_text(yaml.safe_dump(_jsonable(value), allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, dst)
