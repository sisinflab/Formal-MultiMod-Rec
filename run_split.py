import os
import shutil

from elliot.run import run_experiment

import argparse

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--data', type=str, default='baby')
args = parser.parse_args()

if not (os.path.exists(f'./data/{args.data}/train.tsv') and os.path.exists(f'./data/{args.data}/val.tsv') and os.path.exists(f'./data/{args.data}/test.tsv')):
    run_experiment(f"config_files/split.yml")
    shutil.move(f'./data/{args.data}_splits/0/test.tsv', f'./data/{args.data}/test.tsv')
    shutil.move(f'./data/{args.data}_splits/0/0/train.tsv', f'./data/{args.data}/train.tsv')
    shutil.move(f'./data/{args.data}_splits/0/0/val.tsv', f'./data/{args.data}/val.tsv')
    shutil.rmtree(f'./data/{args.data}_splits/')
