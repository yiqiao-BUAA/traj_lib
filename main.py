# traj_lib/main.py
import argparse, importlib, json, sys
from pathlib import Path
from traj_lib.utils.register import DATALOADER_REGISTRY, EVAL_REGISTRY, VIEW_REGISTRY
from traj_lib.utils.exargs   import ConfigResolver
from traj_lib.utils.logger   import get_logger
import traj_lib.utils.dataloader   # 触发注册
import traj_lib.utils.eval         # 触发注册
import traj_lib.utils.views        # 触发注册

log = get_logger('traj_lib.main')

# ---------- CLI ----------
def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="model dir name under traj_lib/model/")
    p.add_argument("--dataset", default="all", help="'all' or comma list of datasets")
    p.add_argument("--metrics", default="all", help="'all' or comma list of metrics")
    p.add_argument("--cfg",     default=None,  help="extra YAML config path")
    return p.parse_args()

# ---------- Config ----------
def load_cfg(dataset, cfg_path):
    if cfg_path:
        return ConfigResolver(cfg_path).parse()
    default = Path(__file__).resolve().parent / "data" / dataset / f"{dataset}.yaml"
    if default.exists():
        return ConfigResolver(str(default)).parse()
    return {}  # 没有 YAML 就返回空 dict

# ---------- 主流程 ----------
def main():
    args = parse_cli()

    # 数据集列表
    datasets = (
        list(DATALOADER_REGISTRY)
        if args.dataset.lower() == "all"
        else [d.strip() for d in args.dataset.split(",")]
    )

    # 评测指标列表
    metric_keys = (
        list(EVAL_REGISTRY)
        if args.metrics.lower() == "all"
        else [m.strip() for m in args.metrics.split(",")]
    )

    # 动态 import 模型
    mod_path = f"traj_lib.model.{args.model}.main"
    try:
        model_mod = importlib.import_module(mod_path)
    except ModuleNotFoundError:
        log.error("Cannot import model module %s", mod_path)
        sys.exit(1)

    if not hasattr(model_mod, "inference"):
        log.error("%s must expose inference()", mod_path)
        sys.exit(1)
    if not hasattr(model_mod, "train"):
        log.warning("%s does not expose train(), only inference() will be used", mod_path)

    model_args = ConfigResolver(f"traj_lib/model/{args.model}/{args.model}.yaml").parse()

    # 逐数据集评测
    for ds in datasets:
        if ds not in DATALOADER_REGISTRY:
            log.error("Dataset '%s' not registered; skip", ds)
            continue
        dataloader = DATALOADER_REGISTRY[ds](
            model_args=model_args,
            pre_views=model_mod.pre_views if hasattr(model_mod, "pre_views") else None,
            post_views=model_mod.post_views if hasattr(model_mod, "post_views") else None
        )
        cfg = load_cfg(ds, args.cfg)

        preds, gts = model_mod.inference(dataloader, **cfg)

        scores = {}
        for m in metric_keys:
            if m not in EVAL_REGISTRY:
                log.error("Metric '%s' not registered; skip", m)
                continue
            score = EVAL_REGISTRY[m](preds, gts)
            scores[m] = score
            log.info("[%s] %-12s : %.6f", ds, m, score)

        # 保存 JSON
        if scores:
            out = {"model": args.model, "dataset": ds, "scores": scores}
            out_dir = Path(__file__).resolve().parent / "outputs"
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"{args.model}_{ds}.json"
            out_file.write_text(json.dumps(out, indent=2))
            log.info("scores saved to %s", out_file)

if __name__ == "__main__":
    main()
