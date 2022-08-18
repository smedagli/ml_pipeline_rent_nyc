#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import tempfile

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(input_args):
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(input_args)

    # Download input artifact.
    # This will also note that this script is using this particular version of the artifact
    logger.info(f"Fetching artifact {input_args.input}")
    artifact_local_path = run.use_artifact(input_args.input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(df, test_size=input_args.test_size, random_state=input_args.random_seed,
                                      stratify=df[input_args.stratify_by] if input_args.stratify_by != 'none' else None)

    # Save to output files
    for df, k in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {k}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w") as fp:
            df.to_csv(fp.name, index=False)
            log_artifact(f"{k}_data.csv", f"{k}_data", f"{k} split of dataset", fp.name, run)


def parse_args():
    parser = argparse.ArgumentParser(description="Split test and remainder")
    parser.add_argument("input", type=str, help="Input artifact to split")
    parser.add_argument("test_size", type=float,
                        help="Size of the test split. Fraction of the dataset, or number of items")
    parser.add_argument("--random_seed", type=int, default=42, required=False,
                        help="Seed for random number generator")
    parser.add_argument("--stratify_by", type=str, default='none', required=False,
                        help="Column to use for stratification")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    go(args)
