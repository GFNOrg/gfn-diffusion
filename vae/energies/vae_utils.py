import torch
import torchvision
from torchvision import datasets
import math

import numpy as np

_EPS = 1.e-5

logtwopi = math.log(2 * math.pi)

_VAE_DATA_PATH = 'energies/data/'
_MNIST_EVALUATION_SUBSET = 'energies/data/mnist_evaluation_subset.npy'


def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    distribution = torch.distributions.Normal(mean, scale)

    log_likelihood = distribution.log_prob(x).sum(1)
    return log_likelihood


def log_normal_diag(x, mu, log_var):
    log_p = -0.5 * (logtwopi + log_var + torch.exp(-log_var) * (x - mu) ** 2.).sum(1)
    return log_p


def log_standard_normal(x):
    log_p = -0.5 * (logtwopi + x ** 2.).sum(1)
    return log_p


def log_bernoulli(x, p):
    pp = torch.clamp(p, _EPS, 1. - _EPS)
    log_p = (x * torch.log(pp) + (1. - x) * torch.log(1. - pp)).sum(1)
    return log_p


def estimate_distribution(model):
    distribution = torch.distributions.MultivariateNormal(torch.zeros(20), torch.eye(20))
    # z_samples = distribution.sample((100000,)).to(device)
    z_samples = distribution.sample((100000,))
    x_prediction_samples = model.decode(z_samples)
    vae_posterior_mu = torch.mean(x_prediction_samples, dim=0)
    vae_posterior_std = torch.std(x_prediction_samples, dim=0)
    vae_posterior_logvar = torch.log(vae_posterior_std ** 2)
    return vae_posterior_mu, vae_posterior_std, vae_posterior_logvar


def get_dataloaders(batch_size, device):

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == torch.device("cuda") else {}

    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(_VAE_DATA_PATH, train=True, download=True,
                       transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(_VAE_DATA_PATH, train=False, transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size, shuffle=False, **kwargs)

    evaluation_subset = torch.from_numpy(np.load(_MNIST_EVALUATION_SUBSET)).to(device)

    return train_dataloader, test_dataloader, evaluation_subset