#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 .gitignore 语法复制目录内容，支持为规则选择匹配基准：
- --config-base source|config  控制规则相对谁解析（默认 source=源目录根）
- 规则文件首行可用 "# base: source|config" 覆盖该文件的基准

特性
- .gitignore 语法（支持 '!' 反排除、顺序覆盖）
- 可指定多个 -c/--config（后者追加在前者之后）
- --dry-run 预览、执行前确认、进度条与“当前复制项”显示
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    import pathspec
except Exception:
    print("缺少依赖 pathspec，请先安装： pip install pathspec tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def read_lines(conf_path: Path):
    if not conf_path.exists():
        print(f"⚠️ 规则文件不存在，已跳过：{conf_path.as_posix()}")
        return []
    text = conf_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    eff = sum(1 for ln in lines if ln.strip() and not ln.lstrip().startswith("#"))
    print(f"📘 读取规则：{conf_path.as_posix()}（有效行 {eff}）")
    return lines


def parse_file_base_override(lines):
    """
    规则文件首行可写: "# base: source" 或 "# base: config"
    返回 None / "source" / "config"
    """
    if not lines:
        return None
    first = lines[0].strip().lower()
    if first.startswith("# base:"):
        v = first.split(":", 1)[1].strip()
        if v in ("source", "config"):
            return v
    return None


def rewrite_patterns(lines, src_root: Path, conf_dir: Path, mode: str):
    """
    将一份规则文件的行，按基准模式改写为“相对源目录根”的等效规则：
    - mode='source' : 不改写（保持针对源根）
    - mode='config' : 将以规则文件所在目录 conf_dir 为根来解释，再改写为源根相对的等效规则
      规则改写要点（保留 .gitignore 语义）：
        * 以 '!' 开头的反排除保留 '!'，对其后的模式做相同改写
        * 以 '/' 开头表示相对“基准根”锚定，去掉前导 '/' 后再拼接 base 前缀
        * 非 '/' 开头视为相对基准根的“任意层级”匹配，拼接前缀 "<base>/" 再保留原模式
    """
    out = []
    # conf 基于 src_root 的相对前缀
    try:
        base_rel = conf_dir.relative_to(src_root).as_posix()
    except Exception:
        # 若规则文件不在源目录树内，则仍按绝对路径折算
        base_rel = conf_dir.as_posix()

    for raw in lines:
        s = raw.rstrip("\r\n")
        if not s or s.lstrip().startswith("#"):
            out.append(s)
            continue

        bang = False
        body = s
        if s.startswith("!"):
            bang = True
            body = s[1:]

        body = body.replace("\\", "/")  # 规范化斜杠

        if mode == "source":
            new_pat = body
        else:
            # mode == "config"
            if body.startswith("/"):
                # 基于 config 目录锚定 -> 去掉开头斜杠，再前置 base_rel
                b = body[1:]
                new_pat = f"{base_rel}/{b}" if base_rel else b
            else:
                # 相对基准根的相对模式 -> 前置 base_rel/
                new_pat = f"{base_rel}/{body}" if base_rel else body

        # 归一化多余的 '/'
        while "//" in new_pat:
            new_pat = new_pat.replace("//", "/")

        out.append(("!" + new_pat) if bang else new_pat)

    return out


def build_spec_from_configs(config_paths, src_root: Path, default_mode: str):
    """
    合并多个规则文件（后者覆盖前者）并返回 PathSpec。
    每个文件可通过首行 "# base: xxx" 覆盖 default_mode。
    """
    patterns = []
    for c in config_paths:
        p = Path(c)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        lines = read_lines(p)
        if not lines:
            continue
        override = parse_file_base_override(lines)
        mode = override or default_mode
        rew = rewrite_patterns(lines, src_root=src_root, conf_dir=p.parent, mode=mode)
        patterns.extend(rew)
    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)


def should_copy(rel_posix: str, is_dir: bool, spec: pathspec.PathSpec) -> bool:
    """
    True => 复制；False => 忽略
    目录在匹配时需要在末尾补 '/' 以适配 'foo/' 类规则
    """
    candidate = rel_posix + ("/" if is_dir and not rel_posix.endswith("/") else "")
    return not spec.match_file(candidate)


def iter_all_entries(src_root: Path):
    """
    枚举 src_root 下的所有路径（目录优先于文件）
    返回 (kind, rel_posix, abs_path)
    """
    root_posix = src_root.as_posix()
    dirs, files = [], []
    for p in src_root.rglob("*"):
        rel = p.as_posix()[len(root_posix) + 1:]
        if not rel:
            continue
        (dirs if p.is_dir() else files).append((rel, p))
    for rel, p in sorted(dirs):
        yield ("dir", rel, p)
    for rel, p in sorted(files):
        yield ("file", rel, p)


def print_plan(plan):
    if not plan:
        print("ℹ️ 计划为空。")
        return
    print(f"✅ 最终计划（共 {len(plan)} 项）：")
    for kind, rel, dest in plan:
        tag = "DIR " if kind == "dir" else "FILE"
        print(f"  [{tag}] {rel}  ->  {dest}")


def main():
    ap = argparse.ArgumentParser(description="用 .gitignore 语法复制内容（可选择规则基准 source/config）")
    ap.add_argument("-dest", help="目标目录")
    ap.add_argument("-s", "--source", default=".", help="源目录（默认：当前目录）")
    ap.add_argument("-c", "--config", action="append",
                    help="规则文件（可多次指定；后者覆盖前者）")
    ap.add_argument("--use-gitignore", action="store_true",
                    help="当未提供 --config 时，使用源目录下的 .gitignore 作为规则")
    ap.add_argument("--config-base", choices=["source", "config"], default="source",
                    help="规则基准：source=相对源目录根（默认），config=相对各规则文件所在目录")
    ap.add_argument("--dry-run", action="store_true", help="仅预览，不实际复制")
    ap.add_argument("--no-progress", action="store_true", help="关闭进度条显示")
    args = ap.parse_args()

    src_root = Path(args.source).resolve()
    dest_root = Path(args.dest).resolve()
    if not src_root.exists() or not src_root.is_dir():
        print(f"❌ 源目录无效：{src_root.as_posix()}")
        sys.exit(1)

    # 决定规则来源
    config_paths = args.config or []
    if not config_paths:
        copyignore = src_root / ".copyignore"
        if copyignore.exists():
            config_paths = [copyignore.as_posix()]
        elif args.use-gitignore and (src_root / ".gitignore").exists():
            config_paths = [(src_root / ".gitignore").as_posix()]

    if config_paths:
        spec = build_spec_from_configs(config_paths, src_root, args.config_base)
        print("📝 使用规则：", ", ".join(config_paths))
        print(f"📐 规则基准：{args.config_base}（可在文件首行用 '# base: source|config' 覆盖）")
    else:
        print("📝 未提供规则文件：将复制全部内容")
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, [])

    # 构建计划
    plan = []
    for kind, rel, abs_p in iter_all_entries(src_root):
        if should_copy(rel, kind == "dir", spec):
            dest_path = (dest_root / rel).as_posix()
            plan.append((kind, rel, dest_path))

    print(f"📂 源目录:  {src_root.as_posix()}")
    print(f"📁 目标目录: {dest_root.as_posix()}")
    print()
    print_plan(plan)

    if args.dry_run:
        print("\n🧪 dry-run：仅预览，不复制。")
        sys.exit(0)

    ans = input("\n是否继续执行复制？(y/N): ").strip().lower()
    if ans != "y":
        print("❌ 已取消。")
        sys.exit(0)

    print("\n🚀 开始复制...")
    dest_root.mkdir(parents=True, exist_ok=True)

    use_bar = (tqdm is not None) and (not args.no_progress)
    bar = tqdm(total=len(plan), unit="item", ncols=100, desc="复制中") if use_bar else None
    try:
        for kind, rel, dest in plan:
            if bar:
                bar.set_description_str(f"{'DIR ' if kind=='dir' else 'FILE'} {rel}")
                if kind == "file":
                    bar.set_postfix_str(f"file={rel}")
                bar.update(1)

            src_p = src_root / rel
            dest_p = Path(dest)
            if kind == "dir":
                dest_p.mkdir(parents=True, exist_ok=True)
            else:
                dest_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_p, dest_p)
    finally:
        if bar:
            try:
                bar.close()
            except Exception:
                pass

    print("🎉 完成！")


if __name__ == "__main__":
    main()
