import typing

import pandas as pd
from flytekit import Resources, dynamic, workflow

try:
    from .house_price_predictor import fit, generate_and_split_data, predict
except ImportError:
    from house_price_predictor import fit, generate_and_split_data, predict

NUM_HOUSES_PER_LOCATION = 1000
COLUMNS = [
    "PRICE",
    "YEAR_BUILT",
    "SQUARE_FEET",
    "NUM_BEDROOMS",
    "NUM_BATHROOMS",
    "LOT_ACRES",
    "GARAGE_SPACES",
]
# initialize location names to predict house prices in these regions.
LOCATIONS = [
    "NewYork_NY",
    "LosAngeles_CA",
    "Chicago_IL",
    "Houston_TX",
    "Dallas_TX",
    "Phoenix_AZ",
    "Philadelphia_PA",
    "SanAntonio_TX",
    "SanDiego_CA",
    "SanFrancisco_CA",
]

dataset = typing.NamedTuple(
    "GenerateSplitDataOutputs",
    train_data=typing.List[pd.DataFrame],
    val_data=typing.List[pd.DataFrame],
    test_data=typing.List[pd.DataFrame],
)

@dynamic(cache=True, cache_version="0.1", limits=Resources(mem="600Mi"))
def generate_and_split_data_multiloc(
    locations: typing.List[str],
    number_of_houses_per_location: int,
    seed: int,
) -> dataset:
    train_sets = []  # create empty lists for train, validation, and test subsets
    val_sets = []
    test_sets = []
    for _ in locations:
        _train, _val, _test = generate_and_split_data(number_of_houses=number_of_houses_per_location, seed=seed)
        train_sets.append(
            _train,
        )
        val_sets.append(
            _val,
        )
        test_sets.append(
            _test,
        )
    # split the dataset into train, validation, and test subsets
    return train_sets, val_sets, test_sets

@dynamic(cache=True, cache_version="0.1", limits=Resources(mem="600Mi"))
def parallel_fit_predict(
    multi_train: typing.List[pd.DataFrame],
    multi_val: typing.List[pd.DataFrame],
    multi_test: typing.List[pd.DataFrame],
) -> typing.List[typing.List[float]]:
    preds = []

    # generate predictions for multiple regions
    for loc, train, val, test in zip(LOCATIONS, multi_train, multi_val, multi_test):
        model = fit(loc=loc, train=train, val=val)
        preds.append(predict(test=test, model_ser=model))

    return preds

@workflow
def multi_region_house_price_prediction_model_trainer(
    seed: int = 7, number_of_houses: int = NUM_HOUSES_PER_LOCATION
) -> typing.List[typing.List[float]]:
    # generate and split the data
    split_data_vals = generate_and_split_data_multiloc(
        locations=LOCATIONS,
        number_of_houses_per_location=number_of_houses,
        seed=seed,
    )

    # fit the XGBoost model for multiple regions in parallel
    # generate predictions for multiple regions
    predictions = parallel_fit_predict(
        multi_train=split_data_vals.train_data,
        multi_val=split_data_vals.val_data,
        multi_test=split_data_vals.test_data,
    )

    return predictions

if __name__ == "__main__":
    print(multi_region_house_price_prediction_model_trainer())