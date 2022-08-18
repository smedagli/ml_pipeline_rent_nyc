import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig

_STEPS = ["download",
          "basic_cleaning",
          "data_check",
          "data_split",
          "train_random_forest",
          ]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _STEPS

    downloaded_filename = 'sample.csv'
    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in active_steps:
            # Download file and load in W&B
            comp_download = f"{config['main']['components_repository']}/get_data"
            args_download = {"sample": config["etl"]["sample"],
                             "artifact_name": downloaded_filename,
                             "artifact_type": "raw_data",
                             "artifact_description": "Raw file as downloaded"
                             }
            _ = mlflow.run(comp_download, version='main', entry_point="main", parameters=args_download)

        if "basic_cleaning" in active_steps:
            comp_clean = os.path.join(hydra.utils.get_original_cwd(), 'src', 'basic_cleaning')
            output_artifact = "clean_sample.csv"

            args_basic_cleaning = {'input_artifact': f"{downloaded_filename}:latest",
                                   'output_artifact': output_artifact,
                                   'output_type': 'clean_sample',
                                   'output_description': 'Data with outliers and null values removed',
                                   'min_price': config['etl']['min_price'],
                                   'max_price': config['etl']['max_price'],
                                   }

            _ = mlflow.run(comp_clean, entry_point='main', parameters=args_basic_cleaning)

        if "data_check" in active_steps:
            comp_check = os.path.join(hydra.utils.get_original_cwd(), 'src', 'data_check')
            args_test = {'csv': 'clean_sample.csv:latest',
                         'ref': 'clean_sample.csv:reference',
                         'kl_threshold': config['data_check']['kl_threshold'],
                         'min_price': config['etl']['min_price'],
                         'max_price': config['etl']['max_price'],
                         }
            _ = mlflow.run(comp_check, entry_point='main', parameters=args_test)

        if "data_split" in active_steps:
            comp_segregation = f"{config['main']['components_repository']}/train_val_test_split"
            args_data_split = {'input': 'clean_sample.csv:latest',
                               'test_size': config['modeling']['test_size'],
                               'random_seed': config['modeling']['random_seed'],
                               'stratify_by': config['modeling']['stratify_by']}

            _ = mlflow.run(comp_segregation, version='main', entry_point='main', parameters=args_data_split)

        if "train_random_forest" in active_steps:
            comp_train = os.path.join(hydra.utils.get_original_cwd(), 'src', 'train_random_forest')

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            args_train = {'rf_config': rf_config,
                          'trainval_artifact': 'trainval_data.csv:latest',
                          'val_size': config['modeling']['val_size'],
                          'random_seed': config['modeling']['random_seed'],
                          'stratify_by': config['modeling']['stratify_by'],
                          'max_tfidf_features': config['modeling']['max_tfidf_features'],
                          'output_artifact': 'random_forest_export',
                          }

            _ = mlflow.run(comp_train, entry_point='main', parameters=args_train)

        if "test_regression_model" in active_steps:
            comp_regression = f"{config['main']['components_repository']}/test_regression_model"

            args_test = {'mlflow_model': 'random_forest_export:prod',
                         'test_dataset': 'test_data.csv:latest'}

            _ = mlflow.run(comp_regression, entry_point='main', version='main', parameters=args_test)


if __name__ == "__main__":
    go()
