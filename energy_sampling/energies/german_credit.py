from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base_set import BaseSet
from .energies_utils import regression_unnorm_log_prob


_CREDIT_DATA_PATH = 'energies/data/german_credit.data'
_CREDIT_SAMPLES_PATH = 'energies/data/german_credit_samples.npy'


class GermanCredit(BaseSet):
    def __init__(
        self,
        dim: int = 25,
        data_path: str | Path = _CREDIT_DATA_PATH,
        sample_path: str | Path = _CREDIT_SAMPLES_PATH,
        use_prior: bool = False,
        prior_scale: float = 10.0,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__(
        )

        self.use_prior = use_prior
        self.prior_scale = prior_scale
        self.device = device

        self.data = torch.ones(dim, dtype=float).to(self.device)
        self.dim = dim
        self.data_ndim = dim

        # initialize data
        data = torch.from_numpy(np.loadtxt(data_path))
        y = (data[:, -1] - 1).to(torch.bool)
        x = data[:, :-1].to(torch.get_default_dtype())
        x /= torch.std(x, dim=0, keepdim=True)
        x = torch.cat([torch.ones_like(y).unsqueeze(-1), x], dim=-1)

        assert x.ndim == 2
        assert x.shape[0] == 1000
        if not self.dim == x.shape[-1]:
            raise ValueError(f"Dimension is {self.dim} but needs to be 25.")

        # self.register_buffer("x", x, persistent=False)
        # self.register_buffer("y", y, persistent=False)
        self.x = x.to(self.device)
        self.y = y.to(self.device)

        # samples
        samples = torch.from_numpy(np.load(sample_path)).to(torch.get_default_dtype())
        # self.register_buffer("samples", samples, persistent=False)
        self.samples = samples.to(self.device)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return regression_unnorm_log_prob(
            x,
            x=self.x,
            y=self.y,
            use_prior=self.use_prior,
            prior_scale=self.prior_scale,
        )

    def energy(self, x):
        return -self.unnorm_log_prob(x)

    def sample(self, batch_size) -> torch.Tensor:
        # ind = torch.randint(len(self.samples), (batch_size,))
        # return self.samples[ind, :]
        return self.samples[:batch_size, ...]