# Effective XGBoost 

This repository contains examples from [Effective XGBoost book](https://store.metasnake.com/xgboost) with the following flavours:

- Uses polars instead of pandas
- Adds optional Sagemaker training

## Development

### Installation

1. Make sure you have [poetry installed](https://python-poetry.org/docs/#installation)
2. Check that you have Python 3.11 active `python --version`
3. Configure poetry to use Python 3.11 `poetry env use $(which python)`
4. Install the project `poetry install`
5. For visualisations, graphviz is needed. See e.g. [dtreeviz guide](https://github.com/parrt/dtreeviz#installation) for installation

### Linting

`poetry run black . && mypy .`

### Run training locally

```
$ poetry run python -m xgboost_book.survey_model --help

Usage: python -m xgboost_book.survey_model [OPTIONS]

Options:
  --model [decision_tree|xgboost]
                                  [required]
  --hypopt_evals INTEGER
  --help                          Show this message and exit.

```

#### Decision tree example:

`poetry run python -m xgboost_book.survey_model --model decision_tree --hypopt_evals 200`

#### XGBoost example:

Model training:

`poetry run python -m xgboost_book.survey_model --model xgboost --hypopt_evals 2000`

Model evaluation:

```
poetry run python -m xgboost_book.survey_model.sagemaker_model_eval \
    --model-path cache/xgboost-model \
    --test-data-path cache/validation.parquet \
    --output-dir cache/ \
    && cat cache/evaluation.json
```

### Run XGBoost training in Amazon Sagemaker

`poetry run python -m xgboost_book.survey_model.sagemaker --sagemaker-role arn:aws:iam::123456789:role/sagemaker-exec-role`

### Tests

What are those?
