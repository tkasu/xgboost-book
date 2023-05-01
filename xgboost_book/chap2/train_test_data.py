from typing import Tuple

import pandas as pd
import polars as pl
from sklearn import model_selection  # type: ignore


def train_test_split(
    raw_df: pl.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_polars, y_polars = split_X_y(raw_df, "Q6")
    X = X_polars.to_pandas(use_pyarrow_extension_array=True)
    y = y_polars.to_pandas(use_pyarrow_extension_array=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, stratify=y
    )
    return X_train, X_test, y_train, y_test


def split_X_y(df: pl.DataFrame, y_col: str) -> Tuple[pl.DataFrame, pl.Series]:
    # fmt: off
    filtered_df = (
        df
        .filter(pl.col("Q3").is_in(pl.Series(["United States of America", "China", "India"])))
        .filter(pl.col("Q6").is_in(pl.Series(["Data Scientist", "Software Engineer"])))
    )
    # fmt: on
    return filtered_df.drop(y_col), filtered_df.select(y_col).to_series()
