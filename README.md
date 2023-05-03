# Effective XGBoost 

This repository contains examples from [Effective XGBoost book](https://store.metasnake.com/xgboost) with the following flavours:

- Uses polars instead of pandas
- TODO use Sagemaker
- TODO AWS CDK deployment

## Development

### Installation

1. Make sure you have [poetry installed](https://python-poetry.org/docs/#installation)
2. Check that you have Python 3.11 active `python --version`
3. Configure poetry to use Python 3.11 `poetry env use $(which python)`
4. Install the project `poetry install`
5. For visualisations, graphviz is needed. See e.g. [dtreeviz guide](https://github.com/parrt/dtreeviz#installation) for installation

### Linting

`poetry run black . && mypy .`

### Run pipelines locally

`poetry run python -m xgboost_book.survey_model`

### Tests

What are those?
