"""Run all A-D experiments; results live separately from the legacy benchmark."""
import argparse
from src.experiment_abcd import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    run(n_jobs=args.n_jobs)
