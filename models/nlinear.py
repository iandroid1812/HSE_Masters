from darts.models import NLinearModel
import torch
import torchmetrics

def nlinear_default(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train=None, target_val=None):
    try:
        model_nlinear_default = NLinearModel.load_from_checkpoint("model_nlinear_default", best=True)
    except:
        model_nlinear_default = NLinearModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            shared_weights=False,
            const_init=True,
            normalize=True,
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=32,
            n_epochs=50,
            model_name='model_nlinear_default',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=not True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_nlinear_default.fit(
            series=target_train,
            val_series=target_val,
            verbose=True
            )
        
    return model_nlinear_default


def nlinear_minmax(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None):
    try:
        model_nlinear_minmax = NLinearModel.load_from_checkpoint("model_nlinear_minmax", best=True)
    except:
        model_nlinear_minmax = NLinearModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            shared_weights=False,
            const_init=True,
            normalize=False,
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=32,
            n_epochs=50,
            model_name='model_nlinear_minmax',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=not True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_nlinear_minmax.fit(
            series=target_train_scaled,
            # past_covariates=past_train,
            # future_covariates=future_train,
            val_series=target_val_scaled,
            # val_past_covariates=past_val,
            # val_future_covariates=future_val,
            verbose=True
            )
    return model_nlinear_minmax


def nlinear_minmax_cov(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_nlinear_minmax_cov = NLinearModel.load_from_checkpoint("model_nlinear_minmax_cov", best=True)
    except:
        model_nlinear_minmax_cov = NLinearModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            shared_weights=False,
            const_init=True,
            normalize=False,
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=32,
            n_epochs=50,
            model_name='model_nlinear_minmax_cov',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_nlinear_minmax_cov.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )
    return model_nlinear_minmax_cov


def nlinear_minmax_sentiment(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_nlinear_minmax_sentiment = NLinearModel.load_from_checkpoint("model_nlinear_minmax_sentiment", best=True)
    except:
        model_nlinear_minmax_sentiment = NLinearModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            shared_weights=False,
            const_init=True,
            normalize=False,
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=32,
            n_epochs=50,
            model_name='model_nlinear_minmax_sentiment',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_nlinear_minmax_sentiment.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )
    return model_nlinear_minmax_sentiment


def nlinear_minmax_sentiment_opt(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_nlinear_minmax_sentiment_opt = NLinearModel.load_from_checkpoint("model_nlinear_minmax_sentiment_opt", best=True)
    except:
        model_nlinear_minmax_sentiment_opt = NLinearModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            shared_weights=False,
            const_init=True,
            normalize=False,
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=32,
            n_epochs=100,
            model_name='model_nlinear_minmax_sentiment_opt',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_nlinear_minmax_sentiment_opt.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )

    return model_nlinear_minmax_sentiment_opt