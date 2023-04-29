from xgboost_book.chap2.extract import extract_and_cache
from xgboost_book.chap2.preprocessing import clean


URL = "https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip"
CACHE_PATH = "cache/kaggle_survey.parquet"
MEMBER_NAME = "multipleChoiceResponses.csv"


def main():
    df = extract_and_cache(URL, CACHE_PATH, MEMBER_NAME)
    df = clean(df)
    print(df.head(100))
    # print(df.columns)


if __name__ == "__main__":
    main()
