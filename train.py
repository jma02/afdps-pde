import os
import numpy as np

import torch

from sde import SubVPSDE, get_score_fn, EulerMaruayamaPredictor, sample_images
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
        return torch.mean(losses)
    
    return loss_fn


if __name__ == '__main__':
    os.makedirs('samples', exist_ok=True)

    config = {
        'beta_min': 0.1,
        'beta_max': 20.0,
        'timesteps': 1000,
        'lr': 2e-4,
        'warmup': 5000,
        'batch_size': 128,
        'epochs': 100,
        'log_freq': 100,
        'num_workers': 2,
        'use_ema': True
    }
    
    device = 'cuda'

    model = Unet().to(device)

    if config['use_ema']:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )

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

            if config['use_ema']:
                ema_model.update_parameters(model)

            if (step + 1) % config['log_freq'] == 0:
                print(f'Step: {step} ({epoch}) | Loss: {loss.item():.5f} | Grad: {grad.item():.5f}')

            step += 1
        
        model.eval()
        with torch.no_grad():
            print(f'Generating samples at epoch {epoch}')
            score_model = model if not config['use_ema'] else ema_model
            score_fn = get_score_fn(sde, score_model)
            gen_x = sample_images(sde, score_fn, (64, 3, 32, 32), predictor=sampler)
            image = make_im_grid(gen_x, (8, 8))
            image.save(f'samples/{epoch}.png')
    
    torch.save(model, 'unet.pt')
    torch.save(ema_model.state_dict(), 'ema_sd.pt')