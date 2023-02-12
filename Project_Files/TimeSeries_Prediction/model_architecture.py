import torch
import torchmetrics

from darts.models import TFTModel
from pytorch_lightning.callbacks import ModelCheckpoint

callbacks = [ModelCheckpoint(
    every_n_epochs=20,
    save_top_k=-1,
    filename='{epoch}-{val_loss:.2f}-{step}'
    )]

class DirectionalLossAll(torch.nn.Module):
    def __init__(self):
        super(DirectionalLossAll, self).__init__()
                
    def forward(self, pred, true):
        a_pred = torch.roll(pred, 1, dims=1)
        b_pred = pred

        a_true = torch.roll(true, 1, dims=1)
        b_true = true

        pred_ret = torch.subtract(b_pred, a_pred)
        true_ret = torch.subtract(b_true, a_true)

        pred_ret[:, 0, :] = 1
        true_ret[:, 0, :] = 1

    
        res = pred_ret * true_ret
        mask = res.le(0)
        
        loss = torch.square(pred - true)
        loss[mask] *= 1000

        return torch.mean(loss)

def tft(name, INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None, epochs=100):
    try:
        model_tft = TFTModel.load_from_checkpoint(name, best=True)
    except:
        model_tft = TFTModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            full_attention=True,
            feed_forward='SwiGLU',
            dropout=0.25,
            hidden_continuous_size=64,
            categorical_embedding_sizes=None,
            add_relative_index=False,
            loss_fn = DirectionalLossAll(),
            likelihood=None,
            norm_type='RMSNorm',
            torch_metrics=torchmetrics.MeanSquaredError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={
                'lr': 3e-5
                },
            lr_scheduler_cls=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_kwargs={
                'step_size': 20,
                'gamma': 0.5
            },
            batch_size=32,
            n_epochs=epochs,
            model_name=name,
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=False,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", 
                "devices": -1, 
                "enable_progress_bar": True,
                "callbacks": callbacks
            }
        )

        model_tft.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )
            
    return model_tft