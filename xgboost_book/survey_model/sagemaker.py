import boto3  # type: ignore
import click
import numpy as np
import polars as pl
import sagemaker  # type: ignore
import time
from time import gmtime, strftime

from xgboost_book.survey_model.converters import pl_from_pandas_zerocopy
from xgboost_book.survey_model.extract import extract_and_cache
from xgboost_book.survey_model.pipeline import survey_pipeline
from xgboost_book.survey_model.preprocessing import clean_y
from xgboost_book.survey_model.train_test_data import pl_train_test_split

URL = "https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip"
CACHE_DIR = "cache"
CACHE_PATH = f"{CACHE_DIR}/kaggle_survey.parquet"
MEMBER_NAME = "multipleChoiceResponses.csv"


def combine_data_for_sagemaker(X: pl.DataFrame, y: np.ndarray) -> pl.DataFrame:
    return X.with_columns([pl_from_pandas_zerocopy(y).alias("role")]).select(  # type: ignore
        "role",  # y_label needs to be the first column for sagemaker XGBoost
        *X.columns,
    )


@click.command()
@click.option("--sagemaker-role", type=str, required=True)
def main(sagemaker_role: str):
    """
    Run XGBoost training in Amazon Sagemaker.

    This is mostly just copy-paste from:
    https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_parquet_input_training.ipynb
    """
    region = boto3.Session().region_name
    bucket = sagemaker.Session().default_bucket()
    prefix = "sagemaker/xgboost_book/survey"

    df = extract_and_cache(URL, CACHE_PATH, MEMBER_NAME)
    X_train, X_test, y_train, y_test = pl_train_test_split(
        df, test_size=0.3, stratify=True
    )

    X_train_cleaned: pl.DataFrame = survey_pipeline.fit_transform(X_train, y_train)
    X_test_cleaned: pl.DataFrame = survey_pipeline.fit_transform(X_test, y_test)
    y_train_cleaned, y_test_cleaned, encoder = clean_y(y_train, y_test)

    training_data = combine_data_for_sagemaker(X_train_cleaned, y_train_cleaned)
    validation_data = combine_data_for_sagemaker(X_test_cleaned, y_test_cleaned)

    training_data_path = f"{CACHE_DIR}/training.parquet"
    validation_data_path = f"{CACHE_DIR}/validation.parquet"

    training_data.write_parquet(training_data_path)
    validation_data.write_parquet(validation_data_path)

    sagemaker.Session().upload_data(
        training_data_path, bucket=bucket, key_prefix=f"{prefix}/training"
    )
    sagemaker.Session().upload_data(
        validation_data_path, bucket=bucket, key_prefix=f"{prefix}/validation"
    )

    container = sagemaker.image_uris.retrieve("xgboost", region, "1.7-1")

    client = boto3.client("sagemaker", region_name=region)

    training_job_name = "xgboost-survey-data-" + strftime(
        "%Y-%m-%d-%H-%M-%S", gmtime()
    )
    print("Training job", training_job_name)

    create_training_params = {
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "Pipe",
        },
        "RoleArn": sagemaker_role,
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/{prefix}/single-xgboost"},
        "EnableManagedSpotTraining": True,
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.large",
            "VolumeSizeInGB": 1,
        },
        "TrainingJobName": training_job_name,
        "HyperParameters": {
            # From local hypopt run
            "colsample_bytree": "0.812",
            "gamma": "0.00038",
            "learning_rate": " 0.39",
            "max_depth": "8",
            "min_child_weight": "1.46",
            "reg_alpha": "9",
            "reg_lambda": "1.93",
            "subsample": "0.53",
            "objective": "binary:logistic",
            # Sagemaker params
            "num_round": "10",
            "verbosity": "2",
        },
        "StoppingCondition": {
            "MaxWaitTimeInSeconds": 7200,
            "MaxRuntimeInSeconds": 7200,
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/{prefix}/training",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
                "CompressionType": "None",
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/{prefix}/validation",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
                "CompressionType": "None",
            },
        ],
    }

    print(
        f"Creating a training job with name: {training_job_name}. It will take between 5 and 6 minutes to complete."
    )
    client.create_training_job(**create_training_params)

    status = client.describe_training_job(TrainingJobName=training_job_name)[
        "TrainingJobStatus"
    ]
    print(status)
    while status != "Completed" and status != "Failed":
        time.sleep(10)
        status = client.describe_training_job(TrainingJobName=training_job_name)[
            "TrainingJobStatus"
        ]
        print(status)


if __name__ == "__main__":
    main()
