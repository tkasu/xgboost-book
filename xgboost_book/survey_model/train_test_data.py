from typing import Tuple, Optional

import polars as pl
from sklearn import model_selection  # type: ignore

from xgboost_book.survey_model.converters import (
    pl_from_pandas_zerocopy,
)


def pl_train_test_split(
    raw_df: pl.DataFrame, test_size: float, stratify: Optional[bool] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    X_polars, y_polars = split_X_y(raw_df, "Q6")
    X = X_polars.to_pandas(use_pyarrow_extension_array=True)
    y = y_polars.to_pandas(use_pyarrow_extension_array=True)
    stratify_col = y if stratify else None
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, stratify=stratify_col
    )
    return (  # type: ignore
        pl_from_pandas_zerocopy(X_train),
        pl_from_pandas_zerocopy(X_test),
        pl_from_pandas_zerocopy(y_train),
        pl_from_pandas_zerocopy(y_test),
    )


def split_X_y(df: pl.DataFrame, y_col: str) -> Tuple[pl.DataFrame, pl.Series]:
    # fmt: off
    filtered_df = (
        df
        .filter(pl.col("Q3").is_in(pl.Series(["United States of America", "China", "India"])))
        .filter(pl.col("Q6").is_in(pl.Series(["Data Scientist", "Software Engineer"])))
    )
    # fmt: on
    return filtered_df.drop(y_col), filtered_df.select(y_col).to_series()
