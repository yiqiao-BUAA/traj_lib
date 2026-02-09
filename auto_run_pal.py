#!/usr/bin/env python3
"""
Auto Run (mini): 支持串行与“共享配置流水线并行”，并额外维护每个 run 的 stdio 日志（stdout+stderr）。

特性：
- 训练代码固定读取 ./model/{model}/{model}.yaml（无额外路径参数）。
- sweep_spec.yaml 中列出要遍历的键值组合；表达式/注释/引号/锚点会被保留（ruamel.yaml）。
- 无论正常结束、报错还是 Ctrl+C，都会自动恢复原配置。
- extra_name 使用 spec 中的 naming 模板；缺省 "{model}-{dataset}-{hash8}"。
- 兼容参数：--model / --dataset / --spec 以及 --model-name / --dataset-name。

并行模式（流水线）前提：
- 你保证训练进程只会在启动后的前 X 分钟读取 ./model/{model}/{model}.yaml
- 本脚本做法：
    写入组合配置 -> 启动训练 -> 保持 X 分钟不恢复 -> 恢复原配置 -> 启动下一组合
  训练进程本体可并行存在，最多 --parallel 个。

日志（额外维护）：
- 本脚本会为每个 run 维护一个 stdio 日志：{stdio_log_dir}/{extra}.stdio.log
- 子程序自己维护的日志完全不受影响。

依赖：
  pip install ruamel.yaml typer[all] tqdm

用法：
  # 串行
  python auto_run_mini.py --model RNN --dataset NYC --spec ./model/RNN/auto_run.yaml

  # 并行（流水线）
  python auto_run_mini.py --model RNN --dataset NYC --spec ./model/RNN/auto_run.yaml \
    --parallel 4 --read-window-min 2.5
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
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Optional, IO

import typer
from ruamel.yaml import YAML
from tqdm import tqdm

# ------------------------- 基础设置 -------------------------
app = typer.Typer(add_completion=False, help="")

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
    buf: List[str] = []
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buf:
                parts.append(("".join(buf), None))
                buf = []
            i += 1
            continue
        if ch == "[":
            key = "".join(buf) if buf else None
            buf = []
            j = path.find("]", i)
            if j == -1:
                raise ValueError(f"Bad path: {path}")
            idx = int(path[i + 1 : j])
            parts.append(((key or ""), idx))
            i = j + 1
            if i < len(path) and path[i] == ".":
                i += 1
            continue
        if ch in _SEG:
            buf.append(ch)
            i += 1
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
            keys.append(str(k))
            values.append(list(v))
        else:
            raise typer.BadParameter(f"sweep.{k} must be a list")
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
    merged: Dict[str, Any] = {}
    merged.update(_flatten(base_doc))
    merged.update(params)

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


def _atomic_write_yaml(doc: Any, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    save_yaml(doc, tmp)
    os.replace(tmp, target)


def _atomic_write_bytes(data: bytes, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def _open_stdio_log(log_path: Path, cmd: List[str]) -> IO[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = log_path.open("w", encoding="utf-8", errors="replace")
    fp.write(f"# start_time: {datetime.now().isoformat(timespec='seconds')}\n")
    fp.write(f"# cwd: {os.getcwd()}\n")
    fp.write(f"# cmd: {' '.join(shlex.quote(x) for x in cmd)}\n")
    fp.write("# ------------------------------------------------------------\n")
    fp.flush()
    return fp


@dataclass
class Job:
    idx: int
    total: int
    params: Dict[str, Any]
    doc: Any
    extra: str
    cmd: List[str]
    stdio_log_path: Path


@dataclass
class RunningJob:
    job: Job
    proc: subprocess.Popen
    log_fp: IO[str]
    start_mono: float
    restore_deadline: float
    restored: bool = False


def _pipeline_parallel_shared_config(
    *,
    repo: Path,
    cfg_path: Path,
    orig_bytes: bytes,
    jobs: List[Job],
    parallel: int,
    read_window_min: float,
    poll_sec: float,
    stop_on_fail: bool,
) -> int:
    if parallel < 2:
        raise ValueError("parallel must be >= 2 for pipeline mode")
    if read_window_min <= 0:
        raise typer.BadParameter("--read-window-min must be > 0 when --parallel > 1")

    pending = deque(jobs)
    running: List[RunningJob] = []
    active_injection: Optional[RunningJob] = None

    failures: List[Tuple[int, int]] = []
    failed_count = 0

    def restore_original() -> None:
        _atomic_write_bytes(orig_bytes, cfg_path)

    def terminate_all() -> None:
        for rj in running:
            try:
                if rj.proc.poll() is None:
                    rj.proc.terminate()
            except Exception:
                pass

    def close_all_logs() -> None:
        for rj in running:
            try:
                rj.log_fp.flush()
                rj.log_fp.close()
            except Exception:
                pass

    def on_signal(signum, frame):  # noqa: ARG001
        tqdm.write(f"Received signal {signum}, terminating children and restoring config...")
        try:
            terminate_all()
        finally:
            try:
                restore_original()
            finally:
                close_all_logs()
        raise typer.Exit(code=130)

    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    pbar = tqdm(total=len(jobs), desc="runs", unit="run", dynamic_ncols=True)

    try:
        tqdm.write(f"Pipeline parallel enabled: parallel={parallel}, read_window_min={read_window_min}")
        while pending or running or active_injection:
            now = time.monotonic()

            # 1) active_injection 过窗口/退出 -> 恢复原配置
            if active_injection is not None:
                rc = active_injection.proc.poll()
                if rc is not None or now >= active_injection.restore_deadline:
                    if not active_injection.restored:
                        restore_original()
                        active_injection.restored = True
                    active_injection = None

            # 2) 回收已完成进程
            for rj in list(running):
                rc = rj.proc.poll()
                if rc is None:
                    continue

                # 极少情况：它仍是 active_injection，先恢复
                if active_injection is rj and not rj.restored:
                    restore_original()
                    rj.restored = True
                    active_injection = None

                running.remove(rj)

                # 收尾写入
                try:
                    rj.log_fp.write("\n# ------------------------------------------------------------\n")
                    rj.log_fp.write(f"# end_time: {datetime.now().isoformat(timespec='seconds')}\n")
                    rj.log_fp.write(f"# return_code: {rc}\n")
                    rj.log_fp.flush()
                    rj.log_fp.close()
                except Exception:
                    pass

                if rc != 0:
                    failures.append((rj.job.idx, rc))
                    failed_count += 1
                    tqdm.write(
                        f"Job {rj.job.idx}/{rj.job.total} exit={rc} extra={rj.job.extra} "
                        f"stdio_log={rj.job.stdio_log_path}"
                    )
                    if stop_on_fail:
                        pbar.set_postfix_str(f"running={len(running)} queued={len(pending)} failed={failed_count}")
                        terminate_all()
                        restore_original()
                        close_all_logs()
                        pbar.close()
                        return rc

                pbar.update(1)
                pbar.set_postfix_str(f"running={len(running)} queued={len(pending)} failed={failed_count}")

            # 3) 启动新 job：running 未满 + 没有 active_injection + pending 非空
            while pending and (len(running) < parallel) and (active_injection is None):
                job = pending.popleft()

                # 覆盖 config（临界区开始）
                _atomic_write_yaml(job.doc, cfg_path)

                log_fp = _open_stdio_log(job.stdio_log_path, job.cmd)

                tqdm.write(
                    f"Start {job.idx}/{job.total} extra={job.extra}\n"
                    f"  stdio_log: {job.stdio_log_path}"
                )

                proc = subprocess.Popen(
                    job.cmd,
                    cwd=str(repo),
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                rj = RunningJob(
                    job=job,
                    proc=proc,
                    log_fp=log_fp,
                    start_mono=time.monotonic(),
                    restore_deadline=time.monotonic() + read_window_min * 60.0,
                    restored=False,
                )
                running.append(rj)
                active_injection = rj  # 在窗口期内占用 config
                pbar.set_postfix_str(f"running={len(running)} queued={len(pending)} failed={failed_count}")

            # 4) 轮询
            if pending or running or active_injection:
                time.sleep(max(0.05, poll_sec))

        restore_original()
        pbar.set_postfix_str(f"running=0 queued=0 failed={failed_count}")
        pbar.close()

        if failures:
            tqdm.write(f"Finished with {len(failures)} failures: {failures}")
            return 1
        tqdm.write("Finished all runs successfully")
        return 0

    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        try:
            restore_original()
        except Exception:
            pass
        try:
            pbar.close()
        except Exception:
            pass


@app.command()
def inject_run(
    model_name: str = typer.Option(..., "--model-name", "--model", "-m", help="model name"),
    dataset_name: str = typer.Option(..., "--dataset-name", "--dataset", "-d", help="dataset name"),
    spec: Path = typer.Option(..., "--spec", "-s", exists=True, help="parameter combination sweep_spec.yaml"),
    parallel: int = typer.Option(1, "--parallel", "-p", min=1, help="max concurrent training processes"),
    read_window_min: float = typer.Option(
        0.0,
        "--read-window-min",
        help="minutes to keep injected config before restoring (required when --parallel > 1)",
    ),
    poll_sec: float = typer.Option(0.5, "--poll-sec", help="poll interval seconds for process manager"),
    clean_logs: bool = typer.Option(True, "--clean-logs/--no-clean-logs", help="delete existing matching logs before start"),
    stop_on_fail: bool = typer.Option(False, "--stop-on-fail", help="stop launching new jobs when any job fails"),
    stdio_log_dir: Path = typer.Option(Path("./logs/stdio"), "--stdio-log-dir", help="directory for per-run stdio logs"),
):
    # 可选：清理你原本的 logs（不影响 stdio_log_dir）
    if clean_logs:
        template = f"{model_name}-{dataset_name}-*.log"
        log_dir = Path("./logs")
        if log_dir.exists():
            for f in log_dir.glob(template):
                try:
                    f.unlink()
                except Exception:
                    pass
        # 也清理本脚本维护的 stdio logs（同样按 extra 的规则匹配）
        if stdio_log_dir.exists():
            for f in stdio_log_dir.glob(f"{model_name}-{dataset_name}-*.stdio.log"):
                try:
                    f.unlink()
                except Exception:
                    pass

    repo = Path(".")
    cfg_path = repo / "model" / model_name / f"{model_name}.yaml"
    if not cfg_path.exists():
        raise typer.BadParameter(f"Cannot find config file: {cfg_path}")

    base_doc = load_yaml(cfg_path)
    plan = _product_from_spec(spec)
    spec_yaml = load_yaml(spec)
    naming_tpl = spec_yaml.get("naming", f"{model_name}-{dataset_name}-{{hash8}}")

    # 预构建 jobs
    jobs: List[Job] = []
    total = len(plan.combos)
    for i, params in enumerate(plan.combos, 1):
        doc = deepcopy(base_doc)
        for k, v in params.items():
            set_by_path(doc, k, _coerce_yaml_value(v))

        extra = render_extra_name(
            naming_tpl,
            base_doc,
            {**params, "model": model_name, "dataset": dataset_name},
        )
        cmd = [sys.executable, "-m", "main", "--model", model_name, "--dataset", dataset_name, "--extra_name", extra]
        stdio_log_path = stdio_log_dir / f"{extra}.stdio.log"
        jobs.append(Job(idx=i, total=total, params=params, doc=doc, extra=extra, cmd=cmd, stdio_log_path=stdio_log_path))

    orig_bytes = cfg_path.read_bytes()

    # ------------------------- 串行模式 -------------------------
    if parallel <= 1:
        def _backup(target: Path) -> Path:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            bak = target.with_suffix(target.suffix + f".bak.{ts}")
            bak.write_bytes(target.read_bytes())
            return bak

        def _restore(target: Path, bak: Path | None):
            if bak and bak.exists():
                os.replace(bak, target)

        state: Dict[str, Any] = {"child": None, "log_fp": None}

        def _on_signal(signum, frame):  # noqa: ARG001
            child = state.get("child")
            try:
                if child is not None:
                    child.terminate()
            except Exception:
                pass
            try:
                fp = state.get("log_fp")
                if fp:
                    try:
                        fp.flush()
                        fp.close()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                _atomic_write_bytes(orig_bytes, cfg_path)
            except Exception:
                pass
            raise typer.Exit(code=130)

        old_int = signal.getsignal(signal.SIGINT)
        old_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        pbar = tqdm(total=len(jobs), desc="runs", unit="run", dynamic_ncols=True)
        failed_count = 0

        try:
            for job in jobs:
                bak = _backup(cfg_path)
                try:
                    _atomic_write_yaml(job.doc, cfg_path)

                    tqdm.write(
                        f"Run {job.idx}/{job.total} extra={job.extra}\n"
                        f"  stdio_log: {job.stdio_log_path}\n"
                        f"  Command: {' '.join(shlex.quote(x) for x in job.cmd)}"
                    )

                    log_fp = _open_stdio_log(job.stdio_log_path, job.cmd)
                    state["log_fp"] = log_fp

                    proc = subprocess.Popen(
                        job.cmd,
                        cwd=str(repo),
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    state["child"] = proc
                    rc = proc.wait()
                    state["child"] = None

                    try:
                        log_fp.write("\n# ------------------------------------------------------------\n")
                        log_fp.write(f"# end_time: {datetime.now().isoformat(timespec='seconds')}\n")
                        log_fp.write(f"# return_code: {rc}\n")
                        log_fp.flush()
                        log_fp.close()
                    except Exception:
                        pass
                    state["log_fp"] = None

                    if rc != 0:
                        failed_count += 1
                        tqdm.write(f"Non-zero exit={rc} extra={job.extra} stdio_log={job.stdio_log_path}")
                        if stop_on_fail:
                            break

                    pbar.update(1)
                    pbar.set_postfix_str(f"failed={failed_count}")

                finally:
                    _restore(cfg_path, bak)
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            _atomic_write_bytes(orig_bytes, cfg_path)
            pbar.close()
            tqdm.write("Finished all runs and restored config.")
        raise typer.Exit(code=0)

    # ------------------------- 并行模式（流水线） -------------------------
    rc = _pipeline_parallel_shared_config(
        repo=repo,
        cfg_path=cfg_path,
        orig_bytes=orig_bytes,
        jobs=jobs,
        parallel=parallel,
        read_window_min=read_window_min,
        poll_sec=poll_sec,
        stop_on_fail=stop_on_fail,
    )
    raise typer.Exit(code=rc)


@app.callback(invoke_without_command=True)
def _default_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is not None:
        return
    args = sys.argv[1:]
    sys.argv = [sys.argv[0], "inject-run", *args]
    app()


if __name__ == "__main__":
    app()
