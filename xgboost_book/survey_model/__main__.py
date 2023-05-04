import dtreeviz  # type: ignore
import hyperopt
import sklearn  # type: ignore
import xgboost as xgb  # type: ignore
from hyperopt import hp

from xgboost_book.survey_model.extract import extract_and_cache
from xgboost_book.survey_model.hyperparameters import (
    hyperparameter_tuning,
    clean_hypopt_output,
)
from xgboost_book.survey_model.pipeline import survey_pipeline
from xgboost_book.survey_model.preprocessing import clean_y
from xgboost_book.survey_model.train_test_data import pl_train_test_split


URL = "https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip"
CACHE_PATH = "cache/kaggle_survey.parquet"
MEMBER_NAME = "multipleChoiceResponses.csv"


def main():
    df = extract_and_cache(URL, CACHE_PATH, MEMBER_NAME)
    X_train, X_test, y_train, y_test = pl_train_test_split(
        df, test_size=0.3, stratify=True
    )

    X_train_cleaned = survey_pipeline.fit_transform(X_train, y_train)
    X_test_cleaned = survey_pipeline.fit_transform(X_test, y_test)

    y_train_cleaned, y_test_cleaned, encoder = clean_y(y_train, y_test)

    model_type, options = (
        sklearn.tree.DecisionTreeClassifier,
        {
            "max_depth": hp.quniform("max_depth", 1, 8, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 40, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 20, 1),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
            "class_weight": hp.choice("class_weight", [None, "balanced"]),
        },
    )
    # model_type, options = (
    #     xgb.XGBClassifier,
    #     {
    #         "max_depth": hp.quniform("max_depth", 1, 8, 1),
    #         "min_child_weight": hp.loguniform("min_child_weight", -2, 3),
    #         "subsample": hp.uniform("subsample", 0.5, 1),
    #         "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    #         "reg_alpha": hp.uniform("reg_alpha", 0, 10),
    #         "reg_lambda": hp.uniform("reg_lambda", 1, 10),
    #         "gamma": hp.loguniform("gamma", -10, 10),
    #         "learning_rate": hp.loguniform("learning_rate", -7, 0),
    #     },
    # )
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
        max_evals=1_000,
        trials=trials,
    )
    best_params = clean_hypopt_output(options, best_space)
    print(best_params)
    model = model_type(**best_params)
    model.fit(X_train_cleaned, y_train_cleaned)

    print(model.get_params())
    print(f"Model score: {model.score(X_test_cleaned, y_test_cleaned)}")

    viz_model = dtreeviz.model(
        model,
        tree_index=0,
        X_train=X_train_cleaned.to_pandas(use_pyarrow_extension_array=True),
        y_train=y_train_cleaned,
        feature_names=X_train_cleaned.columns,
        target_name="Data Scientist",
        class_names=encoder.classes_,
    )
    viz_model.view().show()
    print(viz_model.node_stats(node_id=0))


if __name__ == "__main__":
    main()
