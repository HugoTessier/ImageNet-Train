import torch


def separates_normal_and_norm_params(model):
    params = []
    norm_params = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            norm_params.append(m.weight)
            norm_params.append(m.bias)
        else:
            if hasattr(m, 'weight'):
                params.append(m.weight)
            if hasattr(m, 'bias'):
                params.append(m.bias)
    return params, norm_params
