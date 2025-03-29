import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from normflows.flows.neural_spline.coupling_test import batch_size
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet

_DATA_DIR = 'energies/data'

class FortyGaussianMixture(BaseSet):
    def __init__(self,
                 dim: int = 50,
                 data_dir: str = _DATA_DIR,
                 device: torch.device = torch.device('cpu')
                 ):
        super().__init__()

        if dim in [2, 50, 200]:
            loc = torch.load(f"{data_dir}/GMM40-{dim}d.pt").to(device)
        else:
            loc = (torch.rand((40, dim), generator=generator) - 0.5) * 2 * 40
        scale = torch.ones_like(loc).to(device)
        mixture_weights = torch.ones(loc.shape[0], device=loc.device)
        modes = D.Independent(D.Normal(loc, scale), 1)
        mix = D.Categorical(mixture_weights)
        self.gmm = D.MixtureSameFamily(mix, modes)

        self.data = torch.tensor([0.0])
        self.device = device

        self.data_ndim = dim

    def gt_logz(self):
        return 0.

    def energy(self, x):
        log_prob = self.gmm.log_prob(x)
        return -log_prob

    def sample(self, batch_size):
        samples = self.gmm.sample((batch_size,)).to(self.device)
        return samples

    def viz_pdf(self, fsave="25gmm-density.png"):
        x = torch.linspace(-15, 15, 100).to(self.device)
        y = torch.linspace(-15, 15, 100).to(self.device)
        X, Y = torch.meshgrid(x, y)
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)  # ?

        density = self.unnorm_pdf(x)
        return x, density

    def __getitem__(self, idx):
        del idx
        return self.data[0]
