import pandas as pd
from darts import TimeSeries
from darts.timeseries import concatenate
from .error import error_count
from helper_funcs.inverse import inverse_func


def historical_predictions(model, target, INPUT_CHUNK, OUTPUT_CHUNK, RETRAIN, LAST, \
    save=True, past=None, future=None):
    """Computes and saves the historical forecasts that would have been obtained by this model on 
    (potentially multiple) series in pickle format for later usage. This method uses an expanding 
    training window it repeatedly builds a training set from the beginning of series. 
    It trains the model on the training set, emits a forecast of length equal to forecast_horizon, 
    and then moves the end of the training set forward by stride time steps.

    Parameters
    ----------
    model : darts.models.forecasting object
        trained forecasting model object
    target : TimeSeries, Sequence[TimeSeries] objects
        scaled target values of validation dataset
    INPUT_CHUNK : int
        number of time steps that will be fed to the internal forecasting module
    OUTPUT_CHUNK : int
        number of time steps to be output by the internal forecasting module
    RETRAIN : bool
        whether and/or on which condition to retrain the model before predicting
    LAST : bool
        whether to retain only the last point of each historical forecast. 
        If set to True, the method returns a single TimeSeries containing the successive 
        point forecasts. Otherwise returns a list of historical TimeSeries forecasts
    save : bool
        whether or not to save historical predictions into pickle format
    past : TimeSeries, Sequence[TimeSeries] objects
        one (or a sequence of) past-observed covariate series
    future : TimeSeries, Sequence[TimeSeries] objects
        one (or a sequence of) of future-known covariate series

    Returns
    -------
    TimeSeries object
        historical forecast for the first timeseries in the list
    """

    prefix = "../../"

    # if model name can't be derived from the parameters we have a baseline model in front of us
    try:
        model_name = model.model_params['model_name']
    except KeyError:
        model_name = 'model_baseline_ '
        hist = model.historical_forecasts(target)

        for i in range(len(hist)):
            company = hist[i]
            company = company.pd_dataframe()
            if save:
                company.to_pickle(prefix + f"Project_Files/Preprocessed_Files/historical/{model_name}_{i}.pkl")

        return hist

    try:
        hist = pd.read_pickle(prefix + f"Project_Files/Preprocessed_Files/historical/{model_name}_0.pkl")
        hist = TimeSeries.from_dataframe(hist)
    except:
    
        hist = model.historical_forecasts(
            series=target,
            past_covariates=past,
            future_covariates=future,
            train_length=INPUT_CHUNK + OUTPUT_CHUNK,
            forecast_horizon=OUTPUT_CHUNK,
            stride=OUTPUT_CHUNK,
            retrain=RETRAIN,
            last_points_only=LAST,
            verbose=True
        )
        
        for i in range(len(hist)):
            company = concatenate(hist[i])
            company = company.pd_dataframe()
            if save:
                company.to_pickle(prefix + f"Project_Files/Preprocessed_Files/historical/{model_name}_{i}.pkl")

        # company = concatenate(hist)
        # company.to_pickle(prefix + f"Project_Files/Preprocessed_Files/historical/{model_name}_{0}.pkl")

    return TimeSeries.from_dataframe(pd.read_pickle(prefix + f"Project_Files/Preprocessed_Files/historical/{model_name}_0.pkl"))

def total_validation(hist, name, target, inverse=False, scaler=None):
    """This function is used to calculate error metrics on validation dataset
    and visualize the predicted timeseries using plot function

    Parameters
    ----------
    hist : TimeSeries object
        timeseries containing historical forecast for validation dataset
    name : str
        model name
    target : TimeSeries object
        scaled target values of validation dataset
    inverse : bool
        whether or not we need to inverse transform hist and target series
    scaler: Scaler object
        the scaler used for transformation of the values

    Returns
    -------
    tuple(TimeSeries, pandas DataFrame)
        historical forecast and dataframe containing error metric calculations
    """
    if inverse:
        hist, target = inverse_func(scaler, hist, target)

    hist.plot(label='predict')
    target[0][30:].plot(label='true')

    df = error_count(target[0], hist, name)

    return hist, df

# def display_prediction_part(target, from_, to_, dictionary):
#     target[0][from_ : to_].plot(label='true')
#     for key, value in dictionary.items():
#         value[from_-30 : to_-30].plot(label=key)
#         print(f"{key} errors")
#         error_print(target[0][from_ : to_], value[from_-30 : to_-30])
#         print('----------------------------------------------------------')
#     return

def model_compare(scaler, target, names, idx, range):
    prefix = "../../"

    total = pd.DataFrame([])
    target = target[idx][range[0]+20:range[1]+20]
    _, target = inverse_func(scaler, [], target)
    # target.plot(label='true')
    
    for name in names:
        hist = pd.read_pickle(prefix + f"Project_Files/Preprocessed_Files/historical/{name}_{idx}.pkl")
        if name == 'model_TFT_HRLOVE':
            hist = hist.pd_dataframe()
        hist = TimeSeries.from_dataframe(hist)

        name = name.split('_')

        hist = hist[range[0]:range[1]]
        
        hist, _ = inverse_func(scaler, hist, target)

        # hist.plot(label=name[1] + name[2])

        df = error_count(target, hist, name[1] + '_' + name[2].replace('S1', 'S'))
        # df['Rank'] = 1
        df.insert(0, 'Rank', 1)
        total = pd.concat([total, df])

    print(total)
    return total