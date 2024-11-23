import numpy as np

import torch

from sde import SubVPSDE, get_score_fn
from unet import Unet
from utils import get_loaders


torch.manual_seed(159753)
np.random.seed(159753)


def get_loss_fn(sde: SubVPSDE, eps=1e-5):
    def loss_fn(model, batch):
        score_fn = get_score_fn(sde, model)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)

        losses = torch.square(score * std[:, None, None, None] + z)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)

        return losses
    return loss_fn


if __name__ == '__main__':
    config = {}

    device = 'cuda'

    model = Unet(config).to(device)
    sde = SubVPSDE(config)

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    train_loader, test_loader = get_loaders(config)
    
    loss_fn = get_loss_fn(sde)
    step = 0
    for epoch in range(1, config['epochs'] + 1):
        for x, _ in train_loader:
            x = x.to(device)

            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model, x)
            loss.backward()

            for g in optim.param_groups:
                g['lr'] = config['lr'] * np.minimum(step / config['warmup'], 1.0)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()

            step += 1
        
        if step % config['eval_freq'] == 0:
            with torch.no_grad():
                ...