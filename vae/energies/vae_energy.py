import torch
from PIL import Image

import matplotlib.pyplot as plt
from .vae import VAE
from .vae_utils import log_standard_normal, log_bernoulli, get_dataloaders

# _VAE_MODEL_PATH = '<path to a pretrained VAE model>'

_VAE_MODEL_PATH = 'energies/data/bce_mnist_vae_epochs_100_seed_1.pt'
# _VAE_MODEL_PATH = 'energies/data/mnist_vae_100_epochs_seed_1_mps.pt'


class VAEEnergy():
    def __init__(self, device, dim=20, batch_size=256, is_linear=True):
        super().__init__()

        self.device = device

        self.data = torch.ones(dim, dtype=float).to(self.device)
        self.data_ndim = dim

        train_dataloader, test_dataloader, evaluation_subset = get_dataloaders(batch_size, device)

        self.train_dataloader = train_dataloader

        self.test_dataloader = test_dataloader

        self.evaluation_subset = evaluation_subset.to(self.device)

        self.vae = VAE().to(self.device)
        self.vae.load_state_dict(torch.load(_VAE_MODEL_PATH))

        # setting vae params to not requiring gradient!
        for param in self.vae.parameters():
            param.requires_grad = False

    def log_prob(self, z, x):
        log_prior = log_standard_normal(z)
        x_hat = self.vae.decode(z)
        log_likelihood = log_bernoulli(x, x_hat)

        return - (log_prior + log_likelihood)

    def ndim(self):
        return self.data_ndim

    def energy(self, z, x):
        return - self.log_prob(z, x)

    def log_reward(self, z, x):
        return self.energy(z, x)

    def sample_train_set(self, batch_size):
        real_data, _ = next(iter(self.train_dataloader))
        real_data = real_data.reshape((-1, 784))
        return real_data

    def sample_test_set(self, batch_size):
        real_data, _ = next(iter(self.test_dataloader))
        real_data = real_data.reshape((-1, 784))
        return real_data

    def sample_evaluation_subset(self, batch_size):
        real_data = self.evaluation_subset[:batch_size].reshape((-1, 784))
        return real_data

    def sample(self, batch_size, evaluation=False):
        if evaluation:
            return self.sample_evaluation_subset(batch_size).to(self.device)
        return self.sample_train_set(batch_size).to(self.device)
