# flyte Testing

Testing flyte

# Install

Add flytectl:

```
curl -sL https://ctl.flyte.org/install | bash
chmod +x ./bin/flytectl
sudo mv ./bin/flytectl /usr/bin/
```

Python:

```
poetry install --no-root
poetry shell
```

# Usage

Local usage:

```
pyflyte run pipelines/hello_world.py hello_world_wf
```

Spin up cluster using Docker:

```
flytectl demo start
```

Remote usage:

```
pyflyte run --remote pipelines/hello_world.py hello_world_wf
```

Remote usage with Docker:

```
pyflyte run --remote --image ghcr.io/flyteorg/flytecookbook:pima_diabetes-latest  pipelines/xgboost_regression.py  house_price_predictor_trainer
```
