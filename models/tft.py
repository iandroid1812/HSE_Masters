from darts.models import TFTModel
import torch
import torchmetrics

def tft_sentiment_opt(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_tft_sentiment_opt = TFTModel.load_from_checkpoint("model_tft_sentiment_opt_1", best=True)
    except:
        model_tft_sentiment_opt = TFTModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            hidden_size=16,
            lstm_layers=2,
            num_attention_heads=4,
            full_attention=False,
            feed_forward='GatedResidualNetwork',
            dropout=0.25,
            hidden_continuous_size=16,
            categorical_embedding_sizes=None,
            add_relative_index=False,
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            norm_type='LayerNorm',

            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=64,
            n_epochs=50,
            model_name='model_tft_sentiment_opt_1',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_tft_sentiment_opt.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )
    return model_tft_sentiment_opt