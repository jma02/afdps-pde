import numpy as np

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

from einops import rearrange


unloader = v2.Compose([v2.Lambda(lambda t: (t + 1) * 0.5),
                       v2.Lambda(lambda t: t.permute(0, 2, 3, 1)),
                       v2.Lambda(lambda t: t * 255.)])


def make_im_grid(x0: torch.Tensor, xy: tuple=(1, 10)):
    x, y = xy
    im = unloader(x0.cpu())
    B, C, H, W = x0.shape
    im = rearrange(im, '(x y) h w c -> (x h) (y w) c', x=B//x, y=B//y).numpy().astype(np.uint8)
    im = v2.ToPILImage()(im)
    return im


def get_loaders(config):
    train_transform = v2.Compose([v2.ToImage(),
                                  v2.RandomHorizontalFlip(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])

    test_transform = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])

    train = datasets.CIFAR10('data/', download=True, train=True, transform=train_transform)
    test = datasets.CIFAR10('data/', download=True, train=False, transform=test_transform)

    bs = config['batch_size']
    j = config['num_workers']

    train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=j, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=j, pin_memory=True, drop_last=True)

    return train_loader, test_loader