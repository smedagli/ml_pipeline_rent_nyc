#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(input_args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(input_args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    print(artifact_local_path)

    ######################
    # YOUR CODE HERE     #
    ######################
    # filename = 'clean_sample.csv'
    # df.to_csv(filename, index=False)
    # upload_to_wandb(run, args, filename)


def upload_to_wandb(wandb_run, args, filename: str = 'clean_sample.csv') -> None:
    artifact = wandb.Artifact(args.output_artifact, type=args.output_type, description=args.output_description)
    artifact.add_file(filename)
    wandb_run.log_artifact(artifact)


def parse_args():
    parser = argparse.ArgumentParser(description="A very basic data cleaning")
    parser.add_argument("--input_artifact", type=str, help="The input artifact", required=True)
    parser.add_argument("--output_artifact", type=str, help="The name of the output artifact", required=True)
    parser.add_argument("--output_type", type=str, help="The type of the output artifact", required=True)
    parser.add_argument("--output_description", type=str, help="A description of the output artifact", required=True)
    parser.add_argument("--min_price", type=float, help="The minumum price to consider", required=True)
    parser.add_argument("--max_price", type=float, help="The maximum price to consider", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    go(args)
