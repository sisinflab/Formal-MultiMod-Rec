from elliot.run import run_experiment
import os

if not (os.path.exists('./data/baby/train.tsv') and os.path.exists('./data/baby/val.tsv') and os.path.exists('./data/baby/test.tsv')):
    run_experiment(f"config_files/run_split.yml")
