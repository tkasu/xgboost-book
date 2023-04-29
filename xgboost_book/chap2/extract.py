from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

import requests
import polars as pl


def extract_zip(url: str, member_name: str) -> pl.DataFrame:
    resp = requests.get(url)
    resp.raise_for_status()

    zip = ZipFile(BytesIO(resp.content))
    csv = zip.open(member_name).read()
    df = pl.read_csv(
        csv,
        skip_rows_after_header=1,
        columns=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
            "Q8",
            "Q9",
            "Q16_Part_1",
            "Q16_Part_2",
            "Q16_Part_3",
        ],
        dtypes={
            "Q2": pl.datatypes.Utf8,
            "Q4": pl.datatypes.Utf8,
            "Q5": pl.datatypes.Utf8,
            "Q8": pl.datatypes.Utf8,
            "Q9": pl.datatypes.Utf8,
            "Q16_Part_1": pl.datatypes.Utf8,
            "Q16_Part_2": pl.datatypes.Utf8,
            "Q16_Part_3": pl.datatypes.Utf8,
        },
    )
    return df


def extract_and_cache(url: str, dst: str, member_name: str) -> pl.DataFrame:
    dst = Path(dst)
    if dst.exists():
        df = pl.read_parquet(dst)
    else:
        df = extract_zip(url, member_name)
        df.write_parquet(dst)
    return df


if __name__ == "__main__":
    url = "https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip"
    member_name = "multipleChoiceResponses.csv"
    df = extract_zip(url, member_name)
    print(df.head(5))
