import torch


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