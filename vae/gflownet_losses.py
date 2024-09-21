import torch
from torch.distributions import Normal


def fwd_tb(initial_state, gfn, log_reward_fn, discretizer, exploration_std=None, return_exp=False, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, log_reward_fn, condition=condition, pis=False)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1], condition).detach()

    loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r) ** 2)
    if return_exp:

        return loss.mean(), states, log_pfs, log_pbs, log_r
    else:

        return loss.mean()


def bwd_tb(initial_state, gfn, log_reward_fn, discretizer, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(initial_state, discretizer, exploration_std, log_reward_fn, condition)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1], condition).detach()

    loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r) ** 2)

    return loss.mean()


def fwd_tb_avg(initial_state, gfn, log_reward_fn, discretizer, exploration_std=None, return_exp=False, condition=None):
    states, log_pfs, log_pbs, _ = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, log_reward_fn, condition=condition, pis=False)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1], condition).detach()

    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1))
    if return_exp:
        return 0.5 * (loss ** 2).mean(), states, log_pfs, log_pbs, log_r
    else:
        return 0.5 * (loss ** 2).mean()


def bwd_tb_avg(initial_state, gfn, log_reward_fn, discretizer, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, _ = gfn.get_trajectory_bwd(initial_state, discretizer, exploration_std, log_reward_fn, condition)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1], condition).detach()

    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1))
    return 0.5 * (loss ** 2).mean()


def fwd_tb_avg_cond(initial_state, gfn, log_reward_fn, discretizer, exploration_std=None, return_exp=False, condition=None,
                    repeats=10):
    condition = condition.repeat(repeats, 1)
    initial_state = initial_state.repeat(repeats, 1)

    states, log_pfs, log_pbs, _ = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, log_reward_fn, condition=condition, pis=False)
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1], condition).detach()

    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).view(repeats, -1).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1)).view(repeats, -1)

    if return_exp:
        return 0.5 * (loss ** 2).mean(), states, log_pfs, log_pbs, log_r
    else:
        return 0.5 * (loss ** 2).mean()


def bwd_tb_avg_cond(initial_state, gfn, log_reward_fn, discretizer, exploration_std=None, condition=None, repeats=10):
    condition = condition.repeat(repeats, 1)
    initial_state = initial_state.repeat(repeats, 1)

    states, log_pfs, log_pbs, _ = gfn.get_trajectory_bwd(initial_state, discretizer, exploration_std, log_reward_fn, condition)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1], condition).detach()

    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).view(repeats, -1).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1)).view(repeats, -1)
    return 0.5 * (loss ** 2).mean()


def db(initial_state, gfn, log_reward_fn, discretizer, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, log_reward_fn, condition=condition, pis=False)
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1], condition).detach()

    loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)
    return loss.mean()


def subtb(initial_state, gfn, log_reward_fn, coef_matrix, discretizer, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, log_reward_fn, condition=condition, pis=False)
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1], condition).detach()

    diff_logp = log_pfs - log_pbs
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp),
         diff_logp.cumsum(dim=-1)),
        dim=1)
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
    A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1
    A2 = A2 ** 2
    return torch.stack([torch.triu(A2[i] * coef_matrix, diagonal=1).sum() for i in range(A2.shape[0])]).sum()


def bwd_mle(samples, gfn, log_reward_fn, discretizer, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(samples, discretizer, exploration_std, log_reward_fn, condition)
    loss = -log_pfs.sum(-1)
    return loss.mean()


def pis(initial_state, gfn, log_reward_fn, discretizer, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, log_reward_fn, condition=condition, pis=True)
    with torch.enable_grad():
        log_r = log_reward_fn(states[:, -1], condition)

    normalization_constant = float(1 / initial_state.shape[-1])
    loss = normalization_constant * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_r)
    return loss.mean()