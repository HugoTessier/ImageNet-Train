import os

import torch
from accelerate import Accelerator


class Logger:
    def __init__(self, accelerator: Accelerator, path: str = 'results'):
        self.top1 = []
        self.top5 = []
        self.accelerator = accelerator
        self.path = os.path.join(os.getcwd(), path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    @staticmethod
    def top1_accuracy(output, target):
        pred = output.argmax(dim=1, keepdim=True)
        return pred.eq(target.view_as(pred)).float().mean().view(1)

    @staticmethod
    def top5_accuracy(output, target):
        _, y_pred = output.topk(k=5, dim=1)
        y_pred = y_pred.t()

        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = (y_pred == target_reshaped)

        ind_which_topk_matched_truth = correct[:5]
        flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
        return flattened_indicator_which_topk_matched_truth.float().mean().view(1)

    def measure_metrics(self, output, target):
        with torch.no_grad():
            self.top1.append(self.top1_accuracy(output, target))
            self.top5.append(self.top5_accuracy(output, target))

    def process_test_results(self, epoch):
        if self.accelerator.is_main_process:
            top1 = torch.mean(torch.cat([torch.mean(self.accelerator.gather_for_metrics(i)).view(1)
                                         for i in self.top1])).item()
            top5 = torch.mean(torch.cat([torch.mean(self.accelerator.gather_for_metrics(i)).view(1)
                                         for i in self.top5])).item()
            self.accelerator.print(f'Epoch n°{epoch}: Top-1 {round(top1, 2)}% Top-5 {round(top5, 2)}%')
            with open(os.path.join(self.path, 'results.txt'), 'a') as f:
                f.write(f'{epoch}: {top1} {top5}')
