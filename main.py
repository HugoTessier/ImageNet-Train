import torch

from utils.args import parse_arguments
from utils.checkpoint import Checkpoint
from utils.dataset import load_debug_dataset, load_imagenet
from utils.metrics import Top5Accuracy, Top1Accuracy
from utils.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from utils.train import train_model


def get_dataset(args):
    if args.debug:
        return load_debug_dataset(args)
    else:
        return load_imagenet(args)


def get_model(args):
    if args.model == 'resnet18':
        return resnet18(args.resnet_width, args.first_layer_width).to(args.device)
    elif args.model == 'resnet34':
        return resnet34(args.resnet_width, args.first_layer_width).to(args.device)
    elif args.model == 'resnet50':
        return resnet50(args.resnet_width, args.first_layer_width).to(args.device)
    elif args.model == 'resnet101':
        return resnet101(args.resnet_width, args.first_layer_width).to(args.device)
    elif args.model == 'resnet152':
        return resnet152(args.resnet_width, args.first_layer_width).to(args.device)
    else:
        print('ERROR: wrong model type')
        raise ValueError


def get_scheduler(optim, args):
    if args.scheduler == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epochs // 3, 2 * (args.epochs // 3)])
    elif args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    elif args.scheduler == 'cosinewr':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=args.T_0, T_mult=args.T_mult)
    else:
        print('ERROR: wrong dataset type!')
        raise ValueError


def create_name(args):
    if args.scheduler == 'cosinewr':
        scheduler = f'{args.scheduler}_{args.T_0}_{args.T_mult}'
    else:
        scheduler = args.scheduler

    name = f'{args.model}_{args.resnet_width}_{args.first_layer_width}_{scheduler}_{args.epochs}_{args.wd}_{args.lr}'
    if args.debug:
        name += '_debug'
    return name


if __name__ == '__main__':
    _args = parse_arguments()

    dataset = get_dataset(_args)
    model = get_model(_args)
    _optim = torch.optim.SGD(params=model.parameters(), lr=_args.lr, weight_decay=_args.wd, momentum=0.9,
                             nesterov=False)
    sched = get_scheduler(_optim, _args)
    met = [Top1Accuracy(), Top5Accuracy()]
    crit = torch.nn.CrossEntropyLoss()

    name = create_name(_args)
    check = Checkpoint(model, _optim, sched, _args.device, _args.distributed, _args.checkpoint_path, name)

    print('\n\nTraining')
    train_model(name=name, checkpoint=check, dataset=dataset, epochs=_args.epochs, criterion=crit, metrics=met,
                output_path=_args.results_path, device=_args.device)
