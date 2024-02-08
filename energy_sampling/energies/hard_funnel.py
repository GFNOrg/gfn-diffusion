import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet


class HardFunnel(BaseSet):
    """
    x0 ~ N(0, 3^2), xi | x0 ~ N(0, exp(x0)), i = 1, ..., 9
    """
    def __init__(self, device, dim=10):
        super().__init__()
        # xlim = 0.01 if nmode == 1 else xlim
        self.device = device

        self.data = torch.ones(dim, dtype=float).to(self.device)
        self.data_ndim = dim

        self.dist_dominant = D.Normal(torch.tensor([0.0]).to(self.device), torch.tensor([3.0]).to(self.device))
        self.mean_other = torch.zeros(self.data_ndim - 1).float().to(self.device)
        self.cov_eye = torch.eye(self.data_ndim - 1).float().to(self.device).view(1, self.data_ndim - 1, self.data_ndim - 1)


    def gt_logz(self):
        return 0.

    def energy(self, x):
        return -self.funnel_log_pdf(x)
    
    def funnel_log_pdf(self, x):
        try:
            dominant_x = x[:, 0]
            log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )

            log_sigma = 0.5 * x[:, 0:1]
            sigma2 = torch.exp(x[:, 0:1])
            neg_log_density_other = 0.5 * np.log(2 * np.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
            log_density_other = torch.sum(-neg_log_density_other, dim=-1)
        except:
            import ipdb;
            ipdb.set_trace()
        return log_density_dominant + log_density_other
    
    def sample(self, batch_size):
        dominant_x = self.dist_dominant.sample((batch_size,))  # (B,1)
        x_others = self._dist_other(dominant_x).sample()  # (B, dim-1)
        return torch.hstack([dominant_x, x_others])

    def _dist_other(self, dominant_x):
        variance_other = torch.exp(dominant_x)
        cov_other = variance_other.view(-1, 1, 1) * self.cov_eye
        # use covariance matrix, not std
        return D.multivariate_normal.MultivariateNormal(self.mean_other, cov_other)

    def viz_pdf(self, fsave="density.png", lim=3):
        raise NotImplementedError

    def __getitem__(self, idx):
        del idx
        return self.data[0]
