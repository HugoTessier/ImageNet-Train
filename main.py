import utils.config as c
from utils.trainer import ImageNetTrainer
import torch
import torchvision
from utils.dataset import load_imagenet
from utils.utils import separates_normal_and_norm_params
from utils.ema import ExponentialMovingAverage
from accelerate import Accelerator

if __name__ == '__main__':
    if c.autotuner:
        torch.backends.cudnn.benchmark = True
    accelerator = Accelerator(gradient_accumulation_steps=1)
    accelerator.gradient_accumulation_steps = c.theoretical_batch_size_for_training / (
            accelerator.num_processes * c.train_batch_size)

    model = torchvision.models.resnet50()
    if c.channels_last:
        model = model.to(memory_format=torch.channels_last, non_blocking=True)

    dataset = load_imagenet(traindir=c.traindir,
                            valdir=c.valdir,
                            accelerator=accelerator,
                            train_batch_size=c.train_batch_size,
                            test_batch_size=c.test_batch_size,
                            mixup_alpha=c.mixup_alpha,
                            cutmix_alpha=c.cutmix_alpha,
                            workers=c.workers,
                            val_resize_size=c.val_resize_size,
                            val_crop_size=c.val_crop_size,
                            train_crop_size=c.train_crop_size,
                            random_erase_prob=c.random_erase,
                            ra_reps=c.ra_reps,
                            hflip_prob=c.hflip_prob,
                            pin_memory=c.pin_memory,
                            persistent_workers=c.persistent_workers)
    params, norm_params = separates_normal_and_norm_params(model)
    optimizer = torch.optim.SGD([{"params": params, "weight_decay": c.weight_decay},
                                 {"params": norm_params, "weight_decay": c.norm_weight_decay}],
                                lr=c.lr,
                                momentum=c.momentum,
                                nesterov=c.nesterov)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                         start_factor=c.lr_warmup_decay,
                                                         total_iters=c.lr_warmup_epochs)
    normal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                  T_max=c.epochs - c.lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler,
                                                                             normal_scheduler],
                                                      milestones=[c.lr_warmup_epochs])

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=c.label_smoothing)
    ema = ExponentialMovingAverage(model,
                                   model_ema_steps=c.model_ema_steps,
                                   model_ema_decay=c.model_ema_decay,
                                   device=accelerator.device,
                                   batch_size=c.train_batch_size,
                                   epochs=c.epochs,
                                   world_size=accelerator.num_processes)

    model, optimizer, scheduler, dataset['train'], dataset['test'] = accelerator.prepare(model,
                                                                                         optimizer,
                                                                                         scheduler,
                                                                                         dataset['train'],
                                                                                         dataset['test'])
    trainer = ImageNetTrainer(epochs=c.epochs,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              criterion=criterion,
                              debug=c.debug,
                              accelerator=accelerator,
                              ema=ema,
                              warmup_epochs=c.lr_warmup_epochs,
                              checkpoint_path=c.checkpoint_path,
                              results_path=c.results_path,
                              save_every_n_epochs=c.save_every_n_epochs)

    trainer(model, dataset)
