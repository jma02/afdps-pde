import numpy as np
import torch

from unet import Unet

# Reference: https://github.com/yang-song/score_sde_pytorch
class SubVPSDE:
    def __init__(self, config):
        self.beta_0 = config['beta_min']
        self.beta_T = config['beta_max']
        self.N = config['timesteps']
    
    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_T - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(-2.0 * self.beta_0 * t - (self.beta_T - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion
    
    def marginal_prob(self, x, t):
        log_mean_coeff = (-0.25 * t ** 2 * (self.beta_T - self.beta_0)
                          -0.5 * t * self.beta_0)
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1.0 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)
    
    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse_sde(self, score_fn, x, t):
        drift, diffusion = self.sde(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score
        return drift, diffusion

    def reverse_discretize(self, score_fn, x, t):
        f, G = self.discretize(x, t)
        rev_f = f - G[: None, None, None] ** 2 * 2 * score_fn(x, t)
        rev_G = G
        return rev_f, rev_G


def get_score_fn(sde: SubVPSDE, model: Unet):
    def score_fn(x, t):
        labels = t * 999 # [0, 1] -> [0, 999]
        score = model(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        score = -score / std[:, None, None, None]
        return score
    return score_fn


class EulerMaruayamaPredictor:
    def __init__(self, sde: SubVPSDE, score_fn):
        self.sde = sde
        self.score_fn = score_fn
    
    def update_fn(self, x, t):
        dt = -1. / self.sde.N
        z = torch.randn_like(x)
        drift, diffusion = self.sde.reverse_sde(self.score_fn, x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean
    
    @torch.no_grad()
    def sample(self, shape, eps=1e-3, device='cuda'):
        x = self.sde.prior_sampling(shape)
        timesteps = torch.linspace(self.sde.T, eps, self.sde.N, device=device)

        for i in range(self.sde.N):
            t = torch.ones(shape[0], device=device) * timesteps[i]
            x, x_mean = self.update_fn(x, t)
        
        return x_mean # Still in [-1, 1]