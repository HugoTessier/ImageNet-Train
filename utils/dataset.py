import torch
from torchvision import transforms, datasets


class DebugDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return torch.rand(3, 224, 224), 0


def load_debug_dataset(args):
    train_dataset = DebugDataset(args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4)
    test_dataset = DebugDataset(args.test_batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              shuffle=False, num_workers=4)

    return {'train': train_loader, 'test': test_loader}


def load_imagenet(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageNet(args.dataset_path, split='train', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    test_dataset = datasets.ImageNet(args.dataset_path, split='val', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return {'train': train_loader, 'test': test_loader}
