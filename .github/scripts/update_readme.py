# .github/scripts/update_readme.py
import json, sys, re
from collections import defaultdict
from pathlib import Path

def load_scores(json_paths):
    data = defaultdict(dict)  # {(model, dataset): {metric: score}}
    metrics = set()
    for p in json_paths:
        obj = json.loads(Path(p).read_text())
        key = (obj["model"], obj["dataset"])
        data[key].update(obj["scores"])
        metrics.update(obj["scores"])
    return data, sorted(metrics)

def render_table(data, metrics):
    hdr = ["Model", "Dataset", *metrics]
    lines = ["| " + " | ".join(hdr) + " |",
             "|-" + "-|-".join('-'*len(h) for h in hdr[1:]) + "-|"]
    for (model, ds) in sorted(data):
        row = [model, ds] + [f"{data[(model, ds)].get(m, ''):.4f}" if m in data[(model, ds)] else "" for m in metrics]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"

def update_readme(readme_path, table_md):
    readme = Path(readme_path).read_text().splitlines()
    start_re = re.compile(r"<!-- *BENCHMARK START *-->")
    end_re   = re.compile(r"<!-- *BENCHMARK END *-->")

    try:
        s = next(i for i, l in enumerate(readme) if start_re.match(l))
        e = next(i for i, l in enumerate(readme) if end_re.match(l))
        new = readme[:s+1] + [table_md] + readme[e:]
    except StopIteration:
        # markers not found → append
        new = readme + ["", "<!-- BENCHMARK START -->", table_md, "<!-- BENCHMARK END -->"]
    Path(readme_path).write_text("\n".join(new))
    print(f"README updated with {len(table_md.splitlines())-2} rows.")

if __name__ == "__main__":
    *json_files, readme_md = sys.argv[1:]
    if not json_files:
        sys.exit("❌  need at least one *.json and README path")
    scores, metrics = load_scores(json_files)
    table_md = render_table(scores, metrics)
    update_readme(readme_md, table_md)
