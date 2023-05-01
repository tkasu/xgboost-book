from typing import Optional

import pyarrow as pa
import pandas as pd
import polars as pl
from feature_engine import encoding, imputation  # type: ignore
from sklearn import base, pipeline  # type: ignore

from xgboost_book.chap2.preprocessing import clean


class KaggleSurveyDataCleaner(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, ycol: Optional[str] = None):
        self.ycol = ycol

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Attempt to do this zero copy, from_pandas clones to data as of polars 0.17.10
        X_polars = pl.from_arrow(pa.Table.from_pandas(X))
        return clean(X_polars).to_pandas(use_pyarrow_extension_array=True)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self


survey_pipeline = pipeline.Pipeline(
    [
        ("clean", KaggleSurveyDataCleaner()),
        (
            "categorise",
            encoding.OneHotEncoder(
                top_categories=5, drop_last=True, variables=["Q1", "Q3", "major"]
            ),
        ),
        (
            "num_impute",
            imputation.MeanMedianImputer(
                imputation_method="median", variables=["education", "years_exp"]
            ),
        ),
    ]
)
