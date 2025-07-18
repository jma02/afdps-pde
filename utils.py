import numpy as np

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

from torch.utils.data import TensorDataset

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
    dataset = torch.load('data/curve_vel_b.pt')

    print(f"Train set shape: {dataset['train'].shape}")
    print(f"Validation set shape: {dataset['val'].shape}")
    print(f"Test set shape: {dataset['test'].shape}")

    train_min = dataset['train'].min()
    train_max = dataset['train'].max()
    val_min = dataset['val'].min()
    val_max = dataset['val'].max()

    dataset_train = dataset['train']
    dataset_val = dataset['val']

    dataset_train = 2.0 * (dataset_train - train_min) / (train_max - train_min) - 1.0
    dataset_val = 2.0 * (dataset_val - val_min) / (val_max - val_min) - 1.0


    train = TensorDataset(dataset_train.detach().clone())
    test = TensorDataset(dataset_val.detach().clone())

    bs = config['batch_size']
    j = config['num_workers']

    train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=j, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=j, pin_memory=True, drop_last=True)

    return train_loader, test_loader