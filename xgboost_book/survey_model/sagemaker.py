from time import gmtime, strftime

import boto3  # type: ignore
import click
import numpy as np
import polars as pl
import sagemaker  # type: ignore
from sagemaker import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

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

    This is mostly done based on example:
    https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform_outputs.html
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

    model_output_path = f"s3://{bucket}/{prefix}/xgboost-pipeline"
    training_job_name = "xgboost-survey-data-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    container = sagemaker.image_uris.retrieve("xgboost", region, "1.7-1")
    xgb_training = Estimator(
        image_uri=container,
        role=sagemaker_role,
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=model_output_path,
        base_job_name=training_job_name,
        region=region,
        volume_size=1,
        max_run=7200,
        max_wait=7200,
        use_spot_instances=True,
    )

    hyperparams = {
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
    }
    xgb_training.set_hyperparameters(**hyperparams)

    train_args = xgb_training.fit(
        inputs={
            "train": TrainingInput(
                s3_data=f"s3://{bucket}/{prefix}/training",
                content_type="application/x-parquet",
            ),
            "validation": TrainingInput(
                s3_data=f"s3://{bucket}/{prefix}/validation",
                content_type="application/x-parquet",
            ),
        }
    )

    xgb_train_step = TrainingStep(
        name="SurveyTraining",
        step_args=train_args,
    )

    pipeline_name = "SurveyPipeline"
    pipeline = Pipeline(name=pipeline_name, steps=[xgb_train_step])

    execution = pipeline.start()
    execution.list_steps()
    execution.wait()


if __name__ == "__main__":
    main()
