from typing import Dict, Any, Union, Optional

import hyperopt
import polars as pl
from sklearn.metrics import accuracy_score


def clean_hypopt_space(
    space: Dict[str, Optional[Union[str, int]]]
) -> Dict[str, Optional[Union[str, int]]]:
    int_vals = ["max_depth", "reg_alpha", "min_samples_leaf", "min_samples_split"]
    return {k: (int(val) if k in int_vals else val) for k, val in space.items()}


def clean_hypopt_output(
    options: Dict[str, Any], space: Dict[str, Optional[Union[str, int]]]
) -> Dict[str, Optional[Union[str, int]]]:
    return clean_hypopt_space(hyperopt.space_eval(options, space))


def hyperparameter_tuning(
    model_type,
    space: Dict[str, Union[str, int]],
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> Dict[str, Any]:
    space = clean_hypopt_space(space)
    model = model_type(**space)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
    return {"loss": -score, "status": hyperopt.STATUS_OK, "model": model}
