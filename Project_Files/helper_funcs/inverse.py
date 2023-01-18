def inverse_func(transformer, pred, true):
    """Calculates inverse transformed values for true and prediction series of values 

    Parameters
    ----------
    transformer : Scaler object
    pred : TimeSeries object
        scaled predicted values outputed by the model
    true : TimeSeries object
        scaled true values that was used as model input

    Returns
    -------
    tuple of TimeSeries objects 
        inverse transformed for predicted aand true values
    """
    return transformer.inverse_transform(pred), transformer.inverse_transform(true)