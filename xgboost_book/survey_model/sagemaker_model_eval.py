import json
import pathlib
import tarfile

import click
import polars as pl

import xgboost as xgb


@click.command()
@click.option("--model-path", default="/opt/ml/processing/model")
@click.option("--test-data-path", default="/opt/ml/processing/test")
@click.option("--output-dir", default="/opt/ml/processing/evaluation")
def eval_model(model_path: str, test_data_path: str, output_dir: str):
    if tarfile.is_tarfile(model_path):
        folder = pathlib.Path(model_path).parent
        with tarfile.open(model_path) as tar:
            tar.extractall(path=folder)
        model_path = folder.joinpath("xgboost-model")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    df = pl.read_parquet(test_data_path)
    X_test = df.drop("role")
    y_test = df.select("role").to_series().to_numpy()
    print(f"X_test columns {X_test.columns}")

    score = model.score(X_test, y_test)

    report_dict = {
        "model_metrics": {
            "score": score,
        },
    }
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))


if __name__ == "__main__":
    eval_model()
