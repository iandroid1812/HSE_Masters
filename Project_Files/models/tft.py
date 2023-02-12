from darts.models import TFTModel
import torch
import torch.nn as nn
import torchmetrics

def my_loss(y_pred, y_true):
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]

    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]

    # print(y_true_next.size())
    # print(y_pred_next.size())

    y_true_diff = torch.subtract(y_true_next, y_true_tdy)
    y_pred_diff = torch.subtract(y_pred_next, y_pred_tdy)

    standard = torch.zeros_like(y_pred_diff)

    y_true_move = torch.ge(y_true_diff, standard)
    y_pred_move = torch.ge(y_pred_diff, standard)

    condition = torch.ne(y_true_move, y_pred_move)
    indices = torch.nonzero(condition)
    
    ones = torch.ones_like(indices)
    indices = torch.add(indices, ones)

    direction_loss = torch.autograd.Variable(torch.ones_like(y_pred)).cuda()


    updates = torch.ones_like(indices).type(torch.cuda.FloatTensor)

    alpha = 1000
    updates = torch.multiply(updates, alpha)
    direction_loss[indices]=1000
    loss = torch.mean(torch.multiply(torch.square(y_true - y_pred), direction_loss))

    return loss

class DirectionalLossLast(nn.Module):
    def __init__(self):
        super(DirectionalLossLast, self).__init__()
                
    def forward(self, pred, true):
        pred_1 = pred[:, 0, :]
        pred_5 = pred[:, -1, :]

        true_1 = true[:, 0, :]
        true_5 = true[:, -1, :]

        pred_ret = torch.subtract(pred_5, pred_1)
        true_ret = torch.subtract(true_5, true_1)
    
        res = pred_ret * true_ret
        mask = res.le(0)
        
        loss = torch.mean(torch.square(pred - true), dim=1)
        loss[mask] *= 1000
        return torch.mean(loss)


class DirectionalLossAll(nn.Module):
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

def tft_sentiment_opt(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_tft_sentiment_opt = TFTModel.load_from_checkpoint("model_tft_sentiment_opt_4", best=True)
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
            # loss_fn = my_loss,
            likelihood=None,
            norm_type='LayerNorm',

            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=64,
            n_epochs=50,
            model_name='model_tft_sentiment_opt_4',
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


def tft_custom_loss(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_tft_custom_loss = TFTModel.load_from_checkpoint("model_tft_custom_loss", best=True)
    except:
        model_tft_custom_loss = TFTModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            hidden_size=16,
            lstm_layers=2,
            num_attention_heads=4,
            full_attention=False,
            feed_forward='SwiGLU',
            dropout=0.3,
            hidden_continuous_size=16,
            categorical_embedding_sizes=None,
            add_relative_index=False,
            loss_fn = my_loss,
            likelihood=None,
            norm_type='RMSNorm',

            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={
                'lr': 1e-3
                },
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=64,
            n_epochs=150,
            model_name='model_tft_custom_loss',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_tft_custom_loss.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )
    return model_tft_custom_loss


def tft_set(name, INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
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
            hidden_continuous_size=32,
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
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
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

def tft_ret(name, INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
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
            # feed_forward='SwiGLU',
            dropout=0.25,
            hidden_continuous_size=32,
            categorical_embedding_sizes=None,
            add_relative_index=False,
            loss_fn = nn.BCEWithLogitsLoss(),
            # loss_fn = nn.BCELoss(),
            likelihood=None,
            norm_type='RMSNorm',
            torch_metrics=torchmetrics.Accuracy(task="binary"),
            optimizer_cls=torch.optim.Adam,
            # optimizer_kwargs={
            #     'lr': 3e-5
            #     },
            # lr_scheduler_cls=torch.optim.lr_scheduler.StepLR,
            # lr_scheduler_kwargs={
            #     'step_size': 20,
            #     'gamma': 0.5
            # },
            batch_size=8,
            n_epochs=epochs,
            model_name=name,
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
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