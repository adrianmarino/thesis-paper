from pytorch_common.callbacks import CallbackManager
from pytorch_common.callbacks.output import Logger
from ..fn import Fn


class AutoEncoderTrainer:
    def __init__(self, model):
        self.model = model

    def fit(
        self,
        data_loader,
        loss_fn,
        epochs,
        encoder_optimizer,
        decoder_optimizer,
        callbacks=[Logger()],
        verbose=1,
        extra_ctx={}
    ):
        """
        Train a autoencoder.
        :param data_loader: data_loader with train set.
        :param loss_fn: function to minimize.
        :param epochs: number of epochs to train model.
        :param encoder_optimizer: optimizer used to adjust encoder.
        :param decoder_optimizer: optimizer used to adjust decoder.
        :param callbacks: callback collection. See Callback.
        :param verbose: show/hide logs.
        """
        callback_manager = CallbackManager(
            epochs,
            encoder_optimizer,
            loss_fn,
            self.model,
            callbacks,
            verbose,
            extra_ctx
        )

        for epoch in range(epochs):
            callback_manager.on_epoch_start(epoch)

            train_loss = Fn.train(
                self.model,
                data_loader,
                loss_fn,
                [encoder_optimizer, decoder_optimizer],
                callback_manager.ctx.device
            )

            callback_manager.on_epoch_end(train_loss)

            if callback_manager.break_training():
                break

        return callback_manager.ctx
