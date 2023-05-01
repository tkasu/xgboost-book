from xgboost_book.chap2.extract import extract_and_cache
from xgboost_book.chap2.pipeline import survey_pipeline
from xgboost_book.chap2.train_test_data import train_test_split


URL = "https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip"
CACHE_PATH = "cache/kaggle_survey.parquet"
MEMBER_NAME = "multipleChoiceResponses.csv"


def main():
    df = extract_and_cache(URL, CACHE_PATH, MEMBER_NAME)
    X_train, X_test, y_train, y_test = train_test_split(df)
    X_train_cleaned = survey_pipeline.fit_transform(X_train, y_train)
    X_test_cleaned = survey_pipeline.fit_transform(X_test, y_test)
    print(X_train_cleaned.head(5))
    print(X_test_cleaned.head(5))
    print(y_train.head(5))
    print(y_test.head(5))


if __name__ == "__main__":
    main()
