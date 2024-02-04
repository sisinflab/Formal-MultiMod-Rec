from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--setting', type=str, default='1')
args = parser.parse_args()

run_experiment(f"config_files/baby_{args.setting}.yml")
