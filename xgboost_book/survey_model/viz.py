from typing import Union, Type, List

import dtreeviz  # type: ignore
import polars as pl
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from xgboost import XGBClassifier  # type: ignore


def get_visualisation_model(
    model: Union[XGBClassifier, DecisionTreeClassifier],
    class_names: List[str],
    X_train: pl.DataFrame,
    y_train: pl.Series,
):
    return dtreeviz.model(
        model,
        tree_index=0,
        X_train=X_train.to_pandas(use_pyarrow_extension_array=True),
        y_train=y_train,
        feature_names=X_train.columns,
        target_name="Data Scientist",
        class_names=class_names,
    )
