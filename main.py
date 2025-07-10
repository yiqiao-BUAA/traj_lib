import argparse, importlib, json
from pathlib import Path
from traj_lib.utils.register import DATALOADER_REGISTRY, EVAL_REGISTRY
from traj_lib.utils.exargs   import ConfigResolver
from traj_lib.utils.logger   import get_logger
import traj_lib.utils.dataloader   # 触发注册
import traj_lib.utils.eval         # 触发注册

log = get_logger('traj_lib.main')

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="model directory name under traj_lib/model/")
    p.add_argument("--dataset", required=True, help="dataset key registered in DATALOADER_REGISTRY")
    p.add_argument("--metrics", default="accuracy", help="comma-sep metric keys")
    p.add_argument("--cfg",     default=None, help="extra YAML config path")
    return p.parse_args()

def load_cfg(dataset, cfg_path):
    if cfg_path:
        return ConfigResolver(cfg_path).parse()

    default = (
        Path(__file__).resolve().parent / "data" / dataset / f"{dataset}.yaml"
    )
    return ConfigResolver(str(default)).parse()

def main():
    args = parse_cli()

    if args.dataset not in DATALOADER_REGISTRY:
        raise KeyError(f"Unknown dataset '{args.dataset}'")
    dataloader = DATALOADER_REGISTRY[args.dataset]()

    mod_path = f"traj_lib.model.{args.model}.main"
    model_mod = importlib.import_module(mod_path)
    if not hasattr(model_mod, "train"):
        log.warning("%s does not expose train(), only inference() will be used", mod_path)
    if not hasattr(model_mod, "inference"):
        raise AttributeError(f"{mod_path} must expose inference()")

    cfg = load_cfg(args.dataset, args.cfg)
    preds, gts = model_mod.inference(dataloader, **cfg)

    metric_keys = [m.strip() for m in args.metrics.split(",")]
    scores = {}
    for m in metric_keys:
        if m not in EVAL_REGISTRY:
            raise KeyError(f"Metric '{m}' not registered")
        score = EVAL_REGISTRY[m](preds, gts)
        scores[m] = score
        log.info("metric %-12s : %.6f", m, score)

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "scores": scores,
    }
    out_path = Path(__file__).resolve().parent / 'outputs' / f"{args.model}_{args.dataset}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("scores saved to %s", out_path)

if __name__ == "__main__":
    main()
