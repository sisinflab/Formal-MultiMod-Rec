import os
import shutil

from elliot.run import run_experiment

if not (os.path.exists('./data/baby/train.tsv') and os.path.exists('./data/baby/val.tsv') and os.path.exists('./data/baby/test.tsv')):
    run_experiment(f"config_files/run_split.yml")
    shutil.move('./data/baby_splits/test.tsv', './data/baby/test.tsv')
    shutil.move('./data/baby_splits/0/0/train.tsv', './data/baby/train.tsv')
    shutil.move('./data/baby_splits/0/0/val.tsv', './data/baby/val.tsv')
    shutil.rmtree('./data/baby_splits/')
