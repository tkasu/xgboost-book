from typing import Any, Union, Optional, Mapping

import hyperopt  # type: ignore
import polars as pl
from sklearn.metrics import accuracy_score  # type: ignore


HypOptSpaceType = Mapping[str, Optional[Union[str, int]]]


def clean_hypopt_space(space: HypOptSpaceType) -> HypOptSpaceType:
    int_vals = ["max_depth", "reg_alpha", "min_samples_leaf", "min_samples_split"]

    return {
        k: (int(val) if k in int_vals else val)  # type: ignore
        for k, val in space.items()
    }


def clean_hypopt_output(
    options: Mapping[str, Any], space: HypOptSpaceType
) -> HypOptSpaceType:
    return clean_hypopt_space(hyperopt.space_eval(options, space))


def hyperparameter_tuning(
    model_type,
    space: HypOptSpaceType,
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> HypOptSpaceType:
    space = clean_hypopt_space(space)
    model = model_type(**space)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
    return {"loss": -score, "status": hyperopt.STATUS_OK, "model": model}
