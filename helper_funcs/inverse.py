def inverse_func(transformer, pred, true):
    return transformer.inverse_transform(pred), transformer.inverse_transform(true)