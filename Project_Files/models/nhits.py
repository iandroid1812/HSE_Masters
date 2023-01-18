from darts.models import NHiTSModel
import torch
import torchmetrics

def nhits(name, INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None, EPOCHS=100):
    try:
        model_nhits = NHiTSModel.load_from_checkpoint(name, best=True)
    except:
        model_nhits = NHiTSModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            num_stacks=6,
            num_blocks=3,
            num_layers=4,
            layer_widths=512,
            pooling_kernel_sizes=None,
            n_freq_downsample=None,
            dropout=0.1,
            activation='ReLU',
            MaxPool1d=True,
            # loss_fn = DirectionalLossAll(),
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            torch_metrics=torchmetrics.MeanSquaredError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={
                'lr': 8e-4
                },
            lr_scheduler_cls=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_kwargs={
                'step_size': 40,
                'gamma': 0.4
            },
            batch_size=32,
            n_epochs=EPOCHS,
            model_name=name,
            log_tensorboard=True,
            nr_epochs_val_period=1,
            # force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_nhits.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            verbose=True
            )

    return model_nhits
