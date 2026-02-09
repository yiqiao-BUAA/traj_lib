#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例用法：
    python grid_res.py --model iPCM --dir ./logs

功能：
    - 仅读取指定模型名（如 iPCM）相关的 log 文件；
    - 从每个日志中提取倒数第11行至倒数第4行的指标；
    - 按数据集（NYC/TKY/CA）分别计算参数结构得分并排名；
    - 输出跨数据集平均排名最高的参数组合。
"""

import re
import argparse
from pathlib import Path
from collections import deque, defaultdict
from statistics import mean


weight = {
    'ReCall1': 100,
    'ReCall5': 0.2,
    'ReCall10': 0.1,
    'ReCall20': 0.05,
}

dataset = ['CA']

# 匹配指标行的正则
LINE_RE = re.compile(
    r"""\[\s*(?P<ds>NYC|TKY|CA|brightkite_ca|Foursquare_ca)\s*\]\s*
        (?P<metric>NDCG1|NDCG5|NDCG10|NDCG20|ReCall1|ReCall5|ReCall10|ReCall20)
        \s*:\s*(?P<val>[0-9]*\.?[0-9]+)""",
    re.X
)

# 从文件名提取结构
NAME_RE = re.compile(r"^(?P<prefix>.+?)-(NYC|TKY|CA|brightkite_ca|Foursquare_ca)-(?P<rest>.+)\.log$", re.IGNORECASE)


def tail_block8(path: Path):
    """仅读取倒数第11至倒数第4行（8行）"""
    dq = deque(maxlen=11)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    if len(dq) < 11:
        return []
    return list(dq)[0:8]


def parse_metrics(lines):
    """从8行中解析出数据集与指标字典"""
    ds = None
    metrics = {}
    for line in lines:
        m = LINE_RE.search(line)
        if not m:
            continue
        cur_ds = m.group("ds")
        if ds is None:
            ds = cur_ds
        elif ds != cur_ds:
            return None, {}
        metrics[m.group("metric")] = float(m.group("val"))
    return ds, metrics


def param_key_from_filename(fname: str):
    """去除中间的 -NYC- / -TKY- / -CA- 段得到参数结构键"""
    m = NAME_RE.match(fname)
    if not m:
        return None
    return f"{m.group('prefix')}-{m.group('rest')}"


def rank_with_ties(score_map):
    """对 {key: score} 降序排名，返回 {key: rank}"""
    items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    ranks = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][1] == items[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg_rank
        i = j + 1
    return ranks


def main(model: str, log_dir: Path):
    per_ds_scores = defaultdict(dict)

    for log_path in log_dir.glob("*.log"):
        if model not in log_path.name:
            continue  # 仅选择指定模型
        key = param_key_from_filename(log_path.name)
        if not key:
            continue
        lines = tail_block8(log_path)
        if not lines:
            continue
        ds, metrics = parse_metrics(lines)
        if not ds or not metrics:
            continue
        score = sum(metrics[m] * weight[m] for m in weight if m in metrics)
        per_ds_scores[ds][key] = score

    if not per_ds_scores:
        print(f"未在 {log_dir} 中找到模型 {model} 的有效日志。")
        return

    per_ds_ranks = {ds: rank_with_ties(scores) for ds, scores in per_ds_scores.items()}
    print(per_ds_ranks.keys())

    all_keys = {k for ds_scores in per_ds_scores.values() for k in ds_scores}
    avg_rank = {}
    for key in all_keys:
        ranks = [per_ds_ranks[ds].get(key) for ds in dataset if key in per_ds_ranks[ds]]
        if ranks:
            avg_rank[key] = mean(ranks)

    if not avg_rank:
        print("没有计算出平均名次。")
        return

    best_key, best_avg_rank = min(avg_rank.items(), key=lambda kv: kv[1])

    print(f"\n=== 模型 {model} 各数据集结果 ===")
    for ds in dataset:
        if ds not in per_ds_scores:
            continue
        items = sorted(per_ds_scores[ds].items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n[{ds}] 前5名：")
        for i, (k, s) in enumerate(items[:50], start=1):
            print(f"{i:>2}. rank={per_ds_ranks[ds][k]:>4.1f}  score={s:.6f}  key={k}")

    print("\n=== 跨数据集平均名次排行（前5） ===")
    for i, (k, r) in enumerate(sorted(avg_rank.items(), key=lambda kv: kv[1])[:5], start=1):
        print(f"{i:>2}. avg_rank={r:.2f}  key={k}")

    print("\n=== 平均名次最佳参数结构 ===")
    print(f"模型: {model}")
    print(f"key : {best_key}")
    print(f"avg_rank : {best_avg_rank:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于log文件计算模型参数排名")
    parser.add_argument("--model", required=True, help="模型名称关键字，例如 iPCM")
    parser.add_argument("--dir", required=True, help="日志目录路径")
    args = parser.parse_args()
    main(args.model, Path(args.dir))
