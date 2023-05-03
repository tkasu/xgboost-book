from feature_engine import encoding, imputation  # type: ignore
from sklearn import pipeline  # type: ignore

from xgboost_book.survey_model.converters import (
    PandasToPolarsConverter,
    PolarsToPandasConverter,
)
from xgboost_book.survey_model.preprocessing import KaggleSurveyDataCleaner

survey_pipeline = pipeline.Pipeline(
    [
        ("clean", KaggleSurveyDataCleaner()),
        ("to_pandas", PolarsToPandasConverter()),
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
