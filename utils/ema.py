import torch


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """
    Extracted from https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    """

    def __init__(self,
                 model,
                 model_ema_steps,
                 model_ema_decay,
                 device,
                 batch_size,
                 epochs,
                 world_size):
        # Decay adjustment that aims to keep the decay independent of other hyperparameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = world_size * batch_size * model_ema_steps / epochs
        alpha = 1.0 - model_ema_decay
        alpha = min(1.0, alpha * adjust)
        decay = 1.0 - alpha

        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)
        self.i = 0
        self.model = model
        self.model_ema_steps = model_ema_steps
        self.model_ema_decay = decay
        self.warmup_mode = True

    def __call__(self, model):
        if self.i >= self.model_ema_steps:
            self.i = 0
            self.update_parameters(model)
            if self.warmup_mode:
                self.n_averaged.fill_(0)
        self.i += 1
