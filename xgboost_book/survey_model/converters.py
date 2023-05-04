from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa  # type: ignore
from sklearn import base  # type: ignore


def pl_from_pandas_zerocopy(
    df: Union[pd.Series, pd.DataFrame]
) -> Union[pl.DataFrame, pl.Series]:
    """
    Attempt to do zero copy convert of DataFrame from pandas to polars.
    pl.from_pandas clones to data as of polars 0.17.10
    """
    match df:
        case np.ndarray():
            return pl.from_arrow(pa.array(df))
        case pd.Series():
            return pl.from_arrow(pa.Array.from_pandas(df))
        case pd.DataFrame():
            return pl.from_arrow(pa.Table.from_pandas(df))
        case t:
            raise ValueError(f"Expected Dataframe or series, got {type(t)}")


class PandasToPolarsConverter(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, ycol: Optional[str] = None):
        self.ycol = ycol

    def transform(self, X: pd.DataFrame) -> pl.DataFrame:
        return pl_from_pandas_zerocopy(X)  # type: ignore

    def fit(self, X, y=None):
        return self


class PolarsToPandasConverter(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, ycol: Optional[str] = None):
        self.ycol = ycol

    def transform(self, X: pl.DataFrame) -> pd.DataFrame:
        return X.to_pandas(use_pyarrow_extension_array=True)

    def fit(self, X, y=None):
        return self
