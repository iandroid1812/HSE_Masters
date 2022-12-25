from darts.metrics import mape, mae, r2_score, rmse, smape


def error_print(true_inversed, pred_inversed):
    error_mape = mape(
        actual_series=true_inversed,
        pred_series=pred_inversed,
        intersect=True,
    )

    error_mae = mae(
        actual_series=true_inversed,
        pred_series=pred_inversed,
        intersect=True,
    )

    error_r2 = r2_score(
        actual_series=true_inversed,
        pred_series=pred_inversed,
        intersect=True,
    )

    error_rmse = rmse(
        actual_series=true_inversed,
        pred_series=pred_inversed,
        intersect=True,
    )

    error_smape = smape(
        actual_series=true_inversed,
        pred_series=pred_inversed,
        intersect=True,
    )

    print(
        f"MAPE error is {error_mape:.5f} %",
        f"\nMAE error is {error_mae:.5f} %",
        f"\nR2 error is {error_r2:.5f} %",
        f"\nRMSE error is {error_rmse:.5f} %",
        f"\nSMAPE error is {error_smape:.5f} %"
        )