import torch
from accelerate import Accelerator

from utils.checkpoint_manager import CheckPointManager
from utils.logger import Logger
import time
from datetime import timedelta
import numpy as np


class ImageNetTrainer:
    def __init__(self,
                 epochs,
                 warmup_epochs,
                 optimizer,
                 scheduler,
                 criterion,
                 ema,
                 accelerator: Accelerator,
                 checkpoint_manager=CheckPointManager,
                 logger=Logger,
                 debug=False,
                 channels_last=False,
                 results_path='./results',
                 checkpoint_path='./checkpoint'):
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.current_epoch = 0
        self.debug = debug
        self.logger = logger(accelerator=accelerator, path=results_path)
        self.checkpoint_manager = checkpoint_manager(accelerator=accelerator, path=checkpoint_path)
        self.accelerator = accelerator
        self.ema = ema
        self.channels_last = channels_last
        self.warmup_epochs = warmup_epochs
        self.training_start = None
        self.epoch_start = None
        self.epoch_durations = []

    def __call__(self, model, dataset, reset_training=True):
        if self.accelerator.is_main_process:
            self.accelerator.print('Launching training.')
            self.training_start = time.time()
        while self.current_epoch < self.epochs:
            if self.accelerator.is_main_process:
                self.epoch_start = time.time()
            if self.current_epoch == self.warmup_epochs:
                self.ema.warmup_mode = False
            self.current_epoch = self.checkpoint_manager.step(self.current_epoch)
            dataset['train'].sampler.set_epoch(self.current_epoch)
            self.train_one_epoch(model, dataset)
            self.test_one_epoch(model, dataset)
            self.scheduler.step()
            self.current_epoch += 1
            if self.accelerator.is_main_process:
                self.epoch_durations.append(time.time() - self.epoch_start)
                estimation = np.array(self.epoch_durations).mean() * (self.epochs - self.current_epoch)
                self.accelerator.print(
                    f'Total elapsed time: {timedelta(seconds=time.time() - self.training_start)}, '
                    f'elapsed time during last epoch: {timedelta(seconds=self.epoch_durations[-1])}, '
                    f'estimated remaining time: {timedelta(seconds=estimation)}')

    def train_one_epoch(self, model, dataset):
        model.train()
        for i, (data, target) in enumerate(dataset['train']):
            with self.accelerator.accumulate(model):
                if self.debug:
                    if i != 0:
                        break

                if self.channels_last:
                    data = data.to(memory_format=torch.channels_last, non_blocking=True)
                    target = target.to(memory_format=torch.channels_last, non_blocking=True)
                output = model(data)
                loss = self.criterion(output, target.long())
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema(model)

    def test_one_epoch(self, model, dataset):
        with torch.inference_mode():
            model.eval()
            for i, (data, target) in enumerate(dataset['test']):
                if self.debug:
                    if i != 0:
                        break
                if self.channels_last:
                    data = data.to(memory_format=torch.channels_last, non_blocking=True)
                    target = target.to(memory_format=torch.channels_last, non_blocking=True)

                output = model(data)
                self.logger.measure_metrics(output, target)
            self.logger.process_test_results(self.current_epoch)
