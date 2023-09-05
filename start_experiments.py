from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--config', type=str, default='baby')
args = parser.parse_args()

run_experiment(f"config_files/{args.config}.yml")
