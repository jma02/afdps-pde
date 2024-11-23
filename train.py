import numpy as np

import torch

from sde import SubVPSDE, get_score_fn, EulerMaruayamaPredictor
from unet import Unet
from utils import get_loaders, make_im_grid


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
    config = {
        'beta_min': 0.1,
        'beta_max': 20.0,
        'timesteps': 1000,
        'lr': 2e-4,
        'warmup': 5000,
        'batch_size': 128,
        'epochs': 100,
        'log_freq': 50,
        'eval_freq': 5,
        'num_workers': 2,
    }

    device = 'cuda'

    model = Unet().to(device)
    sde = SubVPSDE(config)
    sampler = EulerMaruayamaPredictor()

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    train_loader, _ = get_loaders(config)
    
    loss_fn = get_loss_fn(sde)
    step = 0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)

            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model, x)
            loss.backward()

            for g in optim.param_groups:
                g['lr'] = config['lr'] * np.minimum(step / config['warmup'], 1.0)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()

            if (step + 1) % config['log_freq'] == 0:
                print(f'Step: {step} ({epoch}) | Loss: {loss.item():.5f} | Grad: {grad.item():.5f}')

            step += 1
        
        if step % config['eval_freq'] == 0:
            model.eval()
            with torch.no_grad():
                print(f'Generating samples at epoch {epoch}')
                
                score_fn = get_score_fn(sde, model)
                gen_x = sampler.sample(sde, score_fn, (64, 3, 32, 32))
                image = make_im_grid(gen_x, (8, 8))
                image.save(f'samples/{epoch}.png')