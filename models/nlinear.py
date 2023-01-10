from darts.models import NLinearModel
import torch
import torchmetrics

def print_size(a, b):
    print(a.size(), b.size())

def my_loss(y_pred, y_true):
    # print_size(y_pred, y_true)

    # y_true_next = y_true[:, -1, 0]
    # y_pred_next = y_pred[:, -1, 0]

    y_true_next = torch.select(y_true, 2, 0)
    y_pred_next = torch.select(y_pred, 2, 0)

    # y_true_tdy = y_true[:, 0, 0]
    # y_pred_tdy = y_pred[:, 0, 0]

    y_true_tdy = torch.select(y_true, 2, -1)
    y_pred_tdy = torch.select(y_pred, 2, -1)

    # print_size(y_true_next, y_true_tdy)

    y_true_diff = torch.subtract(y_true_next, y_true_tdy)
    y_pred_diff = torch.subtract(y_pred_next, y_pred_tdy)

    # print_size(y_true_diff, y_pred_diff)

    standard = torch.zeros_like(y_pred_diff)

    y_true_move = torch.ge(y_true_diff, standard)
    y_pred_move = torch.ge(y_pred_diff, standard)

    # print_size(y_true_move, y_pred_move)

    condition = torch.ne(y_true_move, y_pred_move)
    indices = torch.nonzero(condition)
    
    ones = torch.ones_like(indices)
    indices = torch.add(indices, ones)

    direction_loss = torch.autograd.Variable(torch.ones_like(y_true_next)).cuda()

    # print_size(direction_loss, direction_loss)

    updates = torch.ones_like(indices).type(torch.cuda.FloatTensor)

    alpha = 1000
    updates = torch.multiply(updates, alpha)
    direction_loss[indices] = alpha
    
    loss = torch.mean(torch.multiply(torch.square(torch.subtract(y_true_next, y_pred_next)), direction_loss))
    
    return loss

def mse_loss(output, target):
    loss = torch.mean((output[:, -1, :] - target[:, -1, :])**2)
    return loss

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


def nlinear_myloss(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        # raise KeyError
        model_nlinear_myloss = NLinearModel.load_from_checkpoint("model_nlinear_myloss_2", best=True)
    except:
        model_nlinear_myloss = NLinearModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            shared_weights=False,
            const_init=True,
            normalize=False,
            loss_fn = mse_loss,
            likelihood=None,
            torch_metrics=torchmetrics.MeanAbsoluteError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=32,
            n_epochs=150,
            model_name='model_nlinear_myloss_2',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_nlinear_myloss.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )

    return model_nlinear_myloss



def nlinear_minmax_sentiment_opt_updated(INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_nlinear_minmax_sentiment_opt_updated = NLinearModel.load_from_checkpoint("model_nlinear_minmax_sentiment_opt_updated", best=True)
    except:
        model_nlinear_minmax_sentiment_opt_updated = NLinearModel(
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
            model_name='model_nlinear_minmax_sentiment_opt_updated',
            log_tensorboard=True,
            nr_epochs_val_period=1,
            force_reset=True,
            save_checkpoints=True,
            random_state=RANDOM,
            pl_trainer_kwargs={
                "accelerator": "gpu", "devices": -1, "callbacks": callbacks
            }
        )

        model_nlinear_minmax_sentiment_opt_updated.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )

    return model_nlinear_minmax_sentiment_opt_updated


def nlinear_emb(name, INPUT_CHUNK, OUTPUT_CHUNK, RANDOM, \
    callbacks=[], target_train_scaled=None, target_val_scaled=None, \
        past_train_scaled=None, past_val_scaled=None, future_train=None, \
            future_val=None):
    try:
        model_nlinear = NLinearModel.load_from_checkpoint(name, best=True)
    except:
        model_nlinear = NLinearModel(
            input_chunk_length=INPUT_CHUNK,
            output_chunk_length=OUTPUT_CHUNK,
            shared_weights=False,
            const_init=True,
            normalize=False,
            # loss_fn = torch.nn.L1Loss(),
            loss_fn = torch.nn.MSELoss(),
            likelihood=None,
            torch_metrics=torchmetrics.MeanSquaredError(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            batch_size=32,
            n_epochs=150,
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

        model_nlinear.fit(
            series=target_train_scaled,
            past_covariates=past_train_scaled,
            future_covariates=future_train,
            val_series=target_val_scaled,
            val_past_covariates=past_val_scaled,
            val_future_covariates=future_val,
            verbose=True
            )
    return model_nlinear