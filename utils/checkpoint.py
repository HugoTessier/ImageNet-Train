import os

import torch


class Checkpoint:
    def __init__(self, model, optimizer, scheduler, device, distributed, save_folder, name):
        self.model = model.to(device)
        if distributed:
            self.model = torch.nn.DataParallel(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_folder = save_folder
        self.name = name
        self.device = device

    def save_model(self, path, epoch):
        with open(path, 'wb') as f:
            state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            torch.save({'model': state_dict,
                        'epoch': epoch,
                        'optimizer': self.optimizer.state_dict()}, f)

    def store_model(self, epoch):
        if not os.path.isdir(self.save_folder):
            try:
                os.mkdir(self.save_folder)
            except OSError:
                print(f'Failed to create the folder {self.save_folder}')
            else:
                print(f'Created folder {self.save_folder}')
        path = os.path.join(self.save_folder, self.name + '_model.chk')
        if not os.path.isfile(path):
            self.save_model(path, epoch)
            return epoch
        else:
            with open(path, 'rb') as f:
                checkpoint = torch.load(f)
                for k, v in checkpoint['model'].items():
                    checkpoint['model'][k] = v.to(self.device)
            loaded_epoch = checkpoint['epoch']
            if epoch >= loaded_epoch:
                self.save_model(path, epoch)
                return epoch
            elif epoch < loaded_epoch:
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.last_epoch = loaded_epoch
                return loaded_epoch
            else:
                raise ValueError
