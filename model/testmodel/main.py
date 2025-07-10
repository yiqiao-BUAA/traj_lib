# traj_lib/model/testmodel/main.py
from traj_lib.utils.register import DATALOADER_REGISTRY
from traj_lib.utils.exargs import ConfigResolver
import traj_lib.data
import torch
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Configuration Resolver Example")
    parser.add_argument('--config', type=str, default='b', help='Name to the configuration file')
    args = parser.parse_args()
    if args.config not in DATALOADER_REGISTRY:
        raise ValueError(f"Configuration '{args.config}' not found in registry.")
    args = ConfigResolver(f'./traj_lib/data/{args.config}/{args.config}.yaml').parse()
    return args

if __name__ == "__main__":
    args = parse_args()

    dataset_name = args['dataset']
    if dataset_name in DATALOADER_REGISTRY:
        dataloader = DATALOADER_REGISTRY[dataset_name]()
        for batch in dataloader:
            print("Batch:", batch)
    else:
        print(f"Dataset '{dataset_name}' not registered.")