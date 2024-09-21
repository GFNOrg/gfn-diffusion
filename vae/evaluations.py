import torch
from utils import logmeanexp
from sample_metrics import compute_distribution_distances


@torch.no_grad()
def log_partition_function(initial_state, gfn, dicretizer, log_reward_fn, condition=None, pis=False):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, dicretizer, None, log_reward_fn, condition=condition, pis=pis)
    log_r = log_reward_fn(states[:, -1], condition)
    log_weight = log_r + log_pbs.sum(-1) - log_pfs.sum(-1)

    log_Z = logmeanexp(log_weight)
    log_Z_lb = log_weight.mean()
    log_Z_learned = log_fs[:, 0].mean()

    return states[:, -1], log_Z, log_Z_lb, log_Z_learned


@torch.no_grad()
def mean_log_likelihood(data, gfn, dicretizer, log_reward_fn, num_evals=10, condition=None, energy=None):
    bsz = data.shape[0]
    data = data.unsqueeze(1).repeat(1, num_evals, 1).view(bsz * num_evals, -1)
    if condition is not None:
        condition = condition.unsqueeze(1).repeat(1, num_evals, 1).view(bsz * num_evals, -1)
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(data, dicretizer, None, log_reward_fn, condition, energy)
    log_weight = (log_pfs.sum(-1) - log_pbs.sum(-1)).view(bsz, num_evals, -1)
    return logmeanexp(log_weight, dim=1).mean()


@torch.no_grad()
def get_sample_metrics(samples, gt_samples=None, final_eval=False):
    if gt_samples is None:
        return

    return compute_distribution_distances(samples.unsqueeze(1), gt_samples.unsqueeze(1), final_eval)
