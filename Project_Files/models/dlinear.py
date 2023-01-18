from darts.models import DLinearModel
import torch
import torchmetrics

def dlinear_sentiment(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_dlinear_sentiment = DLinearModel.load_from_checkpoint("model_dlinear_sentiment_1", best=True)
    except:
        print('STARTING TRAIN')
        model_dlinear_sentiment = DLinearModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            shared_weights=False,
            kernel_size=25,
            const_init=False,
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=64,
            n_epochs=150,
            model_name='model_dlinear_sentiment_1',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_dlinear_sentiment.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )

    return model_dlinear_sentiment