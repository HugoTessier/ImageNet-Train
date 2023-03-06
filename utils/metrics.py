import torch


class Top5Accuracy:
    def __init__(self):
        self.name = 'Top5Accuracy'

    def __call__(self, output, target):
        result = 0
        for o, t in zip(output, target):
            result += int(t in torch.argsort(o, descending=True)[:5])
        return result


class Top1Accuracy:
    def __init__(self):
        self.name = 'Top1Accuracy'

    def __call__(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        return pred.eq(target.view_as(pred)).sum().item()
