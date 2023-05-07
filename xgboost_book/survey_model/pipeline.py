from feature_engine import encoding, imputation  # type: ignore
from sklearn import pipeline  # type: ignore

from xgboost_book.survey_model.converters import (
    PandasToPolarsConverter,
    PolarsToPandasConverter,
)
from xgboost_book.survey_model.preprocessing import (
    KaggleSurveyDataCleaner,
    PolarsColOrderer,
)

survey_pipeline = pipeline.Pipeline(
    [
        ("clean", KaggleSurveyDataCleaner()),
        ("to_pandas", PolarsToPandasConverter()),
        (
            "categorise",
            encoding.OneHotEncoder(variables=["Q1", "Q3", "major"]),
        ),
        (
            "num_impute",
            imputation.MeanMedianImputer(
                imputation_method="median", variables=["education", "years_exp"]
            ),
        ),
        ("to_polars", PandasToPolarsConverter()),
        ("order_cols", PolarsColOrderer()),
    ]
)
