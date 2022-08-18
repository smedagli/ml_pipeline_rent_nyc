#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def log_artifact(artifact_name, artifact_type, artifact_description, filename, wandb_run):
    """
    Log the provided filename as an artifact in W&B, and add the artifact path to the MLFlow run
    so it can be retrieved by subsequent steps in a pipeline

    :param artifact_name: name for the artifact
    :param artifact_type: type for the artifact (just a string like "raw_data", "clean_data" and so on)
    :param artifact_description: a brief description of the artifact
    :param filename: local filename for the artifact
    :param wandb_run: current Weights & Biases run
    :return: None
    """
    # Log to W&B
    artifact = wandb.Artifact(
        artifact_name,
        type=artifact_type,
        description=artifact_description,
    )
    artifact.add_file(filename)
    wandb_run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()


def go(input_args):
    run = wandb.init(job_type="download_file")
    run.config.update(input_args)

    logger.info(f"Returning sample {input_args.sample}")
    logger.info(f"Uploading {input_args.artifact_name} to Weights & Biases")
    log_artifact(artifact_name=input_args.artifact_name,
                 artifact_type=input_args.artifact_type,
                 artifact_description=input_args.artifact_description,
                 filename=os.path.join("data", input_args.sample),
                 wandb_run=run)


def parse_args():
    parser = argparse.ArgumentParser(description="Download URL to a local destination")
    parser.add_argument("sample", type=str, help="Name of the sample to download")
    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("artifact_type", type=str, help="Output artifact type.")
    parser.add_argument("artifact_description", type=str, help="A brief description of this artifact")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    go(args)
