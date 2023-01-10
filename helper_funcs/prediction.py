import pandas as pd
from darts import TimeSeries
from darts.timeseries import concatenate
from .error import error_print


def historical_predictions(model, target, INPUT_CHUNK, OUTPUT_CHUNK, RETRAIN, LAST, \
    covariates=True, past=None, future=None):
    try:
        model_name = model.model_params['model_name']
    except KeyError:
        model_name = 'baseline'
    print(model_name)
    try:
        hist = pd.read_pickle(f"Datasets/historical/{model_name}.pkl")
        hist = TimeSeries.from_dataframe(hist)
    except:
        if not covariates:
            hist = model.historical_forecasts(
                series=target[0],
                train_length=INPUT_CHUNK+OUTPUT_CHUNK,
                forecast_horizon=OUTPUT_CHUNK,
                stride=OUTPUT_CHUNK,
                retrain=RETRAIN,
                last_points_only=LAST,
                verbose=True,
            )
        else:
            hist = model.historical_forecasts(
                series=target[0],
                past_covariates=past,
                future_covariates=future,
                train_length=INPUT_CHUNK+OUTPUT_CHUNK,
                forecast_horizon=OUTPUT_CHUNK,
                stride=OUTPUT_CHUNK,
                retrain=RETRAIN,
                last_points_only=LAST,
                verbose=True,
            )

        hist = concatenate(hist)

        hist.pd_dataframe().to_pickle(f"Datasets/historical/{model_name}.pkl")

    return hist


def display_prediction_part(target, from_, to_, dictionary):
    target[0][from_ : to_].plot(label='true')
    for key, value in dictionary.items():
        value[from_-30 : to_-30].plot(label=key)
        print(f"{key} errors")
        error_print(target[0][from_ : to_], value[from_-30 : to_-30])
        print('----------------------------------------------------------')
    return