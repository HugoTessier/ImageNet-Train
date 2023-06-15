import os

import torch
from accelerate import Accelerator


class CheckPointManager:
    def __init__(self, accelerator: Accelerator, path: str = 'checkpoint'):
        self.path = os.path.join(os.getcwd(), path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.accelerator = accelerator

    def save_state(self, current_epoch):
        with open(os.path.join(self.path, 'state.pt'), 'wb') as f:
            torch.save({'current_epoch': current_epoch}, f)
        self.accelerator.save_state(self.path)

    def load_state(self):
        self.accelerator.load_state(self.path)

    def get_stored_current_epoch(self):
        if os.path.exists(os.path.join(self.path, 'state.pt')):
            with open(os.path.join(self.path, 'state.pt'), 'rb') as f:
                return torch.load(f)['current_epoch']
        else:
            return -1

    def step(self, current_epoch):
        if self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            stored_current_epoch = self.get_stored_current_epoch()
            if current_epoch < stored_current_epoch:
                current_epoch = stored_current_epoch
                self.load_state()
            else:
                self.save_state(current_epoch)
            return current_epoch
