from typing import Any, Dict, Tuple, Type, Union

from hyperopt import hp  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from xgboost import XGBClassifier  # type: ignore


def get_model_and_options(
    model_name: str,
) -> Tuple[Union[Type[XGBClassifier], Type[DecisionTreeClassifier]], Dict[str, Any],]:
    match model_name:
        case "decision_tree":
            return decision_tree()
        case "xgboost":
            return xgboost()
        case _:
            raise ValueError(f"Invalid model name: {model_name}")


def xgboost() -> Tuple[Type[XGBClassifier], Dict[str, Any]]:
    return (
        XGBClassifier,
        {
            "max_depth": hp.quniform("max_depth", 1, 8, 1),
            "min_child_weight": hp.loguniform("min_child_weight", -2, 3),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "reg_alpha": hp.uniform("reg_alpha", 0, 10),
            "reg_lambda": hp.uniform("reg_lambda", 1, 10),
            "gamma": hp.loguniform("gamma", -10, 10),
            "learning_rate": hp.loguniform("learning_rate", -7, 0),
        },
    )


def decision_tree() -> Tuple[Type[DecisionTreeClassifier], Dict[str, Any]]:
    return (
        DecisionTreeClassifier,
        {
            "max_depth": hp.quniform("max_depth", 1, 8, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 40, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 20, 1),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
            "class_weight": hp.choice("class_weight", [None, "balanced"]),
        },
    )
