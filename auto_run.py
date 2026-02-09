#!/usr/bin/env python3
"""
Auto Run (mini): 按参数组合串行「覆盖 -> 运行 -> 恢复」

满足你的现状：
- 训练代码固定读取 ./model/{model}/{model}.yaml（无额外路径参数）。
- 支持在 sweep_spec.yaml 中列出要遍历的键值组合；表达式/注释/引号/锚点会被保留。
- 无论正常结束、报错还是 Ctrl+C，都会自动恢复原配置。
- extra_name 使用 spec 中的 naming 模板；若缺省，则为 "{model}-{dataset}-{hash8}"。
- 兼容指令：--model / --dataset / --spec 以及 --model-name / --dataset-name 的别名。

依赖：
  pip install ruamel.yaml typer[all]

用法（两种均可）：
  # 子命令风格
  python auto_run_mini.py inject-run \
    --model RNN \
    --dataset NYC \
    --spec ./model/RNN/auto_run.yaml

  # 单命令风格（无需写 inject-run）
  python auto_run_mini.py \
    --model RNN \
    --dataset NYC \
    --spec ./model/RNN/auto_run.yaml
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import shlex
import signal
import subprocess
import sys
import re
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import typer
from rich.console import Console
from ruamel.yaml import YAML

# ------------------------- 基础设置 -------------------------
app = typer.Typer(add_completion=False, help="")
console = Console()

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=2, offset=2)

# ------------------------- YAML 读写 -------------------------

def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def save_yaml(doc: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(doc, f)


_SEG = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"

def _parse_path(path: str) -> List[Tuple[str | None, int | None]]:
    parts: List[Tuple[str | None, int | None]] = []
    buf = []
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buf:
                parts.append(("".join(buf), None)); buf = []
            i += 1; continue
        if ch == "[":
            key = "".join(buf) if buf else None; buf = []
            j = path.find("]", i)
            if j == -1:
                raise ValueError(f"Bad path: {path}")
            idx = int(path[i+1:j])
            parts.append(((key or ""), idx))
            i = j + 1
            if i < len(path) and path[i] == ".":
                i += 1
            continue
        if ch in _SEG:
            buf.append(ch); i += 1
        else:
            raise ValueError(f"Unsupported char '{ch}' in path: {path}")
    if buf:
        parts.append(("".join(buf), None))
    return parts


def set_by_path(doc: Any, path: str, value: Any) -> None:
    cur = doc
    parts = _parse_path(path)
    for k, idx in parts[:-1]:
        if idx is None:
            if k not in cur or not isinstance(cur[k], (dict, list)):
                cur[k] = {}
            cur = cur[k]
        else:
            if k:
                if k not in cur or not isinstance(cur[k], list):
                    cur[k] = []
                cur = cur[k]
            while len(cur) <= idx:
                cur.append({})
            if not isinstance(cur[idx], (dict, list)):
                cur[idx] = {}
            cur = cur[idx]
    k, idx = parts[-1]
    if idx is None:
        cur[k] = value
    else:
        if k:
            if k not in cur or not isinstance(cur[k], list):
                cur[k] = []
            cur = cur[k]
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = value


@dataclass
class SweepPlan:
    combos: List[Dict[str, Any]]


def _product_from_spec(spec_path: Path) -> SweepPlan:
    spec = load_yaml(spec_path)
    sweep = spec.get("sweep", {}) or {}
    if not isinstance(sweep, dict):
        raise typer.BadParameter('Top-level "sweep" in sweep_spec.yaml must be a mapping')
    keys: List[str] = []
    values: List[Sequence[Any]] = []
    for k, v in sweep.items():
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            keys.append(str(k)); values.append(list(v))
        else:
            raise typer.BadParameter(f'sweep.{k} must be a list')
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)] if keys else [{}]
    return SweepPlan(combos=combos)


def _flatten(doc: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            pk = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, pk))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            pk = f"{prefix}[{i}]"
            out.update(_flatten(v, pk))
    else:
        out[prefix] = doc
    return out


def render_extra_name(template: str, base_doc: Any, params: Mapping[str, Any]) -> str:
    merged = {}; merged.update(_flatten(base_doc)); merged.update(params)
    class _FMT(dict):
        def __missing__(self, key):
            return "NA"
    tpl = template
    if "{hash8}" in tpl:
        h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        tpl = tpl.replace("{hash8}", h)
    safe = _FMT(merged)
    for k in list(safe.keys()):
        if "[" in k:
            safe[k.replace("[", "_").replace("]", "")] = safe[k]
    name = tpl.format_map(safe)
    cleaned = [ch if (str(ch).isalnum() or ch in "._-") else "-" for ch in str(name)]
    return "".join(cleaned)[:200]


def _coerce_yaml_value(v: Any) -> Any:
    import re
    if isinstance(v, tuple):
        return [_coerce_yaml_value(x) for x in v]
    if isinstance(v, list):
        return [_coerce_yaml_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _coerce_yaml_value(val) for k, val in v.items()}
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1]
            if re.fullmatch(r"[\s,0-9eE+\-.]+", inner or ""):
                try:
                    return [float(x) for x in inner.split(",") if x.strip()]
                except Exception:
                    return v
    return v

# ------------------------- 注入-运行-恢复（核心） -------------------------

@app.command()
def inject_run(
    model_name: str = typer.Option(..., "--model-name", "--model", "-m", help="model name"),
    dataset_name: str = typer.Option(..., "--dataset-name", "--dataset", "-d", help="dataset name"),
    spec: Path = typer.Option(..., "--spec", "-s", exists=True, help="parameter combination sweep_spec.yaml"),
):

    template = f'{model_name}-{dataset_name}-*.log'
    log_dir = Path("./logs")
    for f in log_dir.glob(template):
        f.unlink()

    repo = Path(".")
    cfg_path = repo / "model" / model_name / f"{model_name}.yaml"
    if not cfg_path.exists():
        raise typer.BadParameter(f"Cannot find config file: {cfg_path}")

    base_doc = load_yaml(cfg_path)
    plan = _product_from_spec(spec)
    spec_yaml = load_yaml(spec)
    naming_tpl = spec_yaml.get("naming", f"{model_name}-{dataset_name}-{{hash8}}")

    state = {"child": None, "bak": None, "restored": False, "iter": 0}

    def _atomic_write(doc: Any, target: Path):
        tmp = target.with_suffix(target.suffix + ".tmp")
        save_yaml(doc, tmp)
        os.replace(tmp, target)

    def _backup(target: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = target.with_suffix(target.suffix + f".bak.{ts}")
        bak.write_bytes(target.read_bytes())
        return bak

    def _restore(target: Path, bak: Path | None):
        if state["restored"]:
            return
        try:
            if bak and bak.exists():
                os.replace(bak, target)
        finally:
            state["restored"] = True

    def _on_signal(signum, frame):  # noqa: ARG001
        child = state.get("child")
        try:
            if child is not None:
                child.terminate()
        except Exception:
            pass
        _restore(cfg_path, state.get("bak"))
        raise typer.Exit(code=130)

    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        for i, params in enumerate(plan.combos, 1):
            state["restored"] = False
            state["iter"] = i
            console.rule(f"Run {i}/{len(plan.combos)}")
            doc = deepcopy(base_doc)
            for k, v in params.items():
                set_by_path(doc, k, _coerce_yaml_value(v))
            bak = _backup(cfg_path); state["bak"] = bak
            try:
                _atomic_write(doc, cfg_path)
                extra = render_extra_name(naming_tpl, base_doc, {**params, "model": model_name, "dataset": dataset_name})
                cmd = [sys.executable, "-m", "main", "--model", model_name, "--dataset", dataset_name, "--extra_name", extra]
                console.print("Command:", " ".join(shlex.quote(x) for x in cmd))
                proc = subprocess.Popen(cmd, cwd=str(repo))
                state["child"] = proc
                rc = proc.wait()
                state["child"] = None
                if rc != 0:
                    console.print(f"[red]Quit due to non-zero exit code {rc}[/red]")
                else:
                    console.print("[green]Run completed successfully[/green]")
            finally:
                _restore(cfg_path, bak)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        console.print("Finished all runs and restored config.")

@app.callback(invoke_without_command=True)
def _default_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is not None:
        return
    args = sys.argv[1:]
    new_argv = [sys.argv[0], "inject-run", *args]
    sys.argv = new_argv
    app()

if __name__ == "__main__":
    app()
