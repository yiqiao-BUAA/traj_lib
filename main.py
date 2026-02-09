# traj_lib/main.py
import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path
import traceback
from collections import defaultdict
import inspect

import pandas as pd

from utils.logger import get_logger, set_model_name, set_dataset_name, set_log_file_name
from utils.register import DATALOADER_REGISTRY, EVAL_REGISTRY, EARLY_STOP_REGISTRY
from utils.exargs import ConfigResolver, ParseDict


# ---------- CLI ----------
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", required=True, help="model dir name under traj_lib/model/"
    )
    p.add_argument("--dataset", default="all", help="'all' or comma list of datasets")
    p.add_argument("--metrics", default="all", help="'all' or comma list of metrics")
    p.add_argument("--cfg", default=None, help="extra YAML config path")
    p.add_argument(
        "--task", default="NPP", help="task name, e.g., NPP, Recoverym2m, etc."
    )
    p.add_argument("--extra_name", default="", help="extra name tag for logging")
    p.add_argument("--save_model", default=True, help="whether to save trained model")
    return p.parse_args()

args = parse_cli()

set_model_name(args.model)          # setting model name
set_dataset_name(args.dataset)      # setting dataset name
set_log_file_name(args.extra_name)  # setting extra name tag
log = get_logger(__name__)

import utils.dataloader # start trigger registration
import utils.eval       # start trigger registration
import utils.views      # start trigger registration
import utils.early_stop # start trigger registration
from utils.export import export_alt

utils.dataloader.register_all(task=args.task)   # register all data loaders
utils.eval.register_all(task=args.task)         # register all evaluation metrics
utils.early_stop.register_all(task=args.task)   # register all early stopping methods


# ---------- Config ----------
def load_cfg(dataset: str, cfg_path: str | None) -> ParseDict:
    if cfg_path:
        return ConfigResolver(cfg_path).parse()
    default = Path(__file__).resolve().parent / "data" / dataset / f"{dataset}.yaml"
    if default.exists():
        return ConfigResolver(str(default)).parse()
    else:
        raise FileNotFoundError(f"No config file found for dataset {dataset} in path {default}")


# ---------- Main Process ----------
def main() -> None:

    #set all seed to 42
    import random
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # dataset list
    datasets = (
        list(DATALOADER_REGISTRY)
        if args.dataset.lower() == "all"
        else [d.strip() for d in args.dataset.split(",")]
    )

    # eval metrics list
    metric_keys = (
        list(EVAL_REGISTRY)
        if args.metrics.lower() == "all"
        else [m.strip() for m in args.metrics.split(",")]
    )

    # dynamic import model
    mod_path = f"model.{args.model}.main"
    try:
        model_mod = importlib.import_module(mod_path)
    except ModuleNotFoundError:
        log.error("Cannot import model module %s", mod_path)
        log.error("Exception: %s", sys.exc_info())
        log.error("Error information: %s", traceback.format_exc())
        sys.exit(1)

    if not hasattr(model_mod, "inference"):
        log.error("%s must expose inference()", mod_path)
        sys.exit(1)
    if not hasattr(model_mod, "model"):
        log.error("%s must expose model", mod_path)
        sys.exit(1)

    train_flag = hasattr(model_mod, "train")
    if not train_flag:
        log.warning(
            "%s does not expose train(), only inference() will be used", mod_path
        )

    model_args = ConfigResolver(f"model/{args.model}/{args.model}.yaml").parse()
    log.info("Model args: %s", model_args)

    # evaluate for each dataset
    for ds in datasets:
        if hasattr(model_mod, "pre_views") and model_mod.pre_views:
            log.info("Using pre-views: %s", model_mod.pre_views)
        else:
            log.info("No pre-views defined for model %s", args.model)
        if hasattr(model_mod, "post_views") and model_mod.post_views:
            log.info("Using post-views: %s", model_mod.post_views)
        else:
            log.info("No post-views defined for model %s", args.model)
        dataloader = DATALOADER_REGISTRY[ds](
            model_args=model_args,
            pre_views=model_mod.pre_views if hasattr(model_mod, "pre_views") else None,
            post_views=(
                model_mod.post_views if hasattr(model_mod, "post_views") else None
            ),
        )


        model_saved = None
        if "val_while_train" in model_args and model_args["val_while_train"]:
            record = []
            log.info("Model will validate while train")
            if not hasattr(model_mod, "train") or not hasattr(model_mod, "inference"):
                raise NotImplementedError(
                    "Model must implement 'train', 'inference' for validation while training model"
                )
            eval_funcs = {}
            for m in metric_keys:
                eval_funcs[m] = EVAL_REGISTRY[m]
            
            if 'patience' not in model_args:
                log.info('No patience found in model_args, will not use early stopping')
            else:
                log.info('Patience found in model_args, will use early stopping')
            
                best_scores = {}
                patience = model_args['patience']
                no_improve_epochs = 0
                
                if not hasattr(model_mod, 'early_stop_func'):
                    log.info('Using default early stopping function')
                    early_stop_func = EARLY_STOP_REGISTRY['default']
                else:
                    log.info(f'Using early stopping function: {model_mod.early_stop_func}')
                    early_stop_func = EARLY_STOP_REGISTRY[model_mod.early_stop_func]

            for res_list in model_mod.train(
                dataloader.train_dataloader,
                dataloader.val_dataloader,
                dataloader.view_value,
                eval_funcs=eval_funcs,
            ):
                if record == []:
                    record = [defaultdict(list) for _ in range(len(res_list))]
                    for i in range(0, len(res_list)):
                        for key, value in res_list[i].items():
                            if key == 'title':
                                record[i][key] = value
                            else:
                                record[i][key] = []
                
                for i in range(0, len(res_list)):
                    for key, value in res_list[i].items():
                        if key == 'title':
                            continue
                        record[i][key].append(value)
                        log.info("[%s] %-12s : %.6f", ds, key, value)

                if 'patience' not in model_args:
                    continue

                res = res_list[0]
                improved, best_scores = early_stop_func(res, best_scores)
                if improved:
                    no_improve_epochs = 0
                    model_saved = getattr(model_mod, 'model')
                else:
                    no_improve_epochs += 1
                    log.info(f'No improvement in {no_improve_epochs:>3d}/{patience:>3d} epochs')
                if no_improve_epochs >= patience:
                    log.info(f'Early stopping triggered after {patience} epochs without improvement.')
                    model_mod.model = model_saved
                    break
    
        else:
            log.info("Model will not validate while train")
            if hasattr(model_mod, "train"):
                record = model_mod.train(
                    dataloader.train_dataloader,
                    dataloader.val_dataloader,
                    dataloader.view_value,
                )

        if inspect.isgeneratorfunction(model_mod.inference):
            log.info("Using generator style inference for evaluation")
            scores, cnt = {}, 0
            for inference_res in model_mod.inference(
                dataloader.test_dataloader,
                dataloader.view_value,
            ):
                preds = inference_res['pred']
                gts = inference_res['gts']
                for m in metric_keys:
                    if m not in scores:
                        scores[m] = 0
                    scores[m] += EVAL_REGISTRY[m](preds, gts) * len(gts)
                cnt += len(gts)
            for key, value in scores.items():
                scores[key] = value / cnt
            log.info(f"total count for evaluation: {cnt}")
        else:
            log.info("Using normal style inference for evaluation")
            scores = {}
            inference_res = model_mod.inference(
                dataloader.test_dataloader,
                dataloader.view_value,
            )
            preds = inference_res['pred']
            gts = inference_res['gts']
            for m in metric_keys:
                score = EVAL_REGISTRY[m](preds, gts)
                scores[m] = score

        log.info('-'*60)
        log.info("Results for dataset %s:", ds)
        for key, score in scores.items():
            log.info("[%s] %-12s : %.6f", ds, key, score)
        log.info('-'*60)

        # export_alt(
        #     record,
        #     title=f"{args.model} on {ds}",
        #     path=f"./logs/{args.model}_{ds}.html")

        # save results to outputs/{model}_{dataset}.json
        if scores:
            out = {"model": args.model, "dataset": ds, "scores": scores}
            out_dir = Path(__file__).resolve().parent / "outputs"
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"{args.model}_{ds}.json"
            out_file.write_text(json.dumps(out, indent=2))
            log.info("scores saved to %s", out_file)

        if args.save_model:
            if model_saved is not None:
                save_dir = Path(__file__).resolve().parent / "saved_models" / f"{ds}"
                save_dir.mkdir(exist_ok=True)
                pickle_path = save_dir / f"{args.model}.pkl"
                pickle.dump(model_saved, open(pickle_path, "wb"))
                log.info("Model %s saved to %s", args.model, pickle_path)
            elif hasattr(model_mod, "model"):
                save_dir = Path(__file__).resolve().parent / "saved_models" / f"{ds}"
                save_dir.mkdir(exist_ok=True)
                pickle_path = save_dir / f"{args.model}.pkl"
                pickle.dump(model_mod.model, open(pickle_path, "wb"))
                log.info("Model %s saved to %s", args.model, pickle_path)
            else:
                log.warning(
                    "Model %s has no attribute 'model', cannot be saved", args.model
                )
        else:
            log.info("Model saving skipped as --save_model not set")


if __name__ == "__main__":
    main()
