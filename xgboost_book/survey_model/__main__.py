import click
import hyperopt  # type: ignore

from xgboost_book.survey_model.extract import extract_and_cache
from xgboost_book.survey_model.hyperparameters import (
    hyperparameter_tuning,
    clean_hypopt_output,
)
from xgboost_book.survey_model.models import get_model_and_options
from xgboost_book.survey_model.pipeline import survey_pipeline
from xgboost_book.survey_model.preprocessing import clean_y
from xgboost_book.survey_model.train_test_data import pl_train_test_split
from xgboost_book.survey_model.viz import get_visualisation_model

URL = "https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip"
CACHE_PATH = "cache/kaggle_survey.parquet"
MEMBER_NAME = "multipleChoiceResponses.csv"


@click.command()
@click.option("--model", type=click.Choice(["decision_tree", "xgboost"]), required=True)
@click.option("--hypopt_evals", type=int, default=100)
def main(model: str, hypopt_evals: int):
    df = extract_and_cache(URL, CACHE_PATH, MEMBER_NAME)
    X_train, X_test, y_train, y_test = pl_train_test_split(
        df, test_size=0.3, stratify=True
    )

    X_train_cleaned = survey_pipeline.fit_transform(X_train, y_train)
    X_test_cleaned = survey_pipeline.fit_transform(X_test, y_test)

    y_train_cleaned, y_test_cleaned, encoder = clean_y(y_train, y_test)

    model_type, options = get_model_and_options(model)
    trials = hyperopt.Trials()

    best_space = hyperopt.fmin(
        fn=lambda space: hyperparameter_tuning(
            model_type,
            space,
            X_train_cleaned,
            y_train_cleaned,
            X_test_cleaned,
            y_test_cleaned,
        ),
        space=options,
        algo=hyperopt.tpe.suggest,
        max_evals=hypopt_evals,
        trials=trials,
    )
    best_params = clean_hypopt_output(options, best_space)
    print(best_params)
    model = model_type(**best_params)
    model.fit(X_train_cleaned, y_train_cleaned)

    print(model.get_params())
    print(f"Model score: {model.score(X_test_cleaned, y_test_cleaned)}")

    viz_model = get_visualisation_model(
        model=model,
        class_names=encoder.classes_,
        X_train=X_train_cleaned,
        y_train=y_train_cleaned,
    )
    viz_model.view().show()
    print(viz_model.node_stats(node_id=0))


if __name__ == "__main__":
    main()
