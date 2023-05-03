import dtreeviz  # type: ignore
import sklearn  # type: ignore
import xgboost as xgb  # type: ignore

from xgboost_book.survey_model.extract import extract_and_cache
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

    #model = xgb.XGBClassifier()
    model = sklearn.tree.DecisionTreeClassifier(max_depth=5)
    model.fit(X_train_cleaned, y_train_cleaned)
    print(model.get_params())
    print(f"Model score: {model.score(X_test_cleaned, y_test_cleaned)}")

    viz_model = dtreeviz.model(
        model,
        tree_index=90,
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
