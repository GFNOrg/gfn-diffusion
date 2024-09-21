import torch
import random
import numpy as np
import math
import PIL

from gflownet_losses import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def dcp(tensor):
    return tensor.detach().cpu()


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def fig_to_image(fig):
    fig.canvas.draw()

    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def get_gfn_optimizer(gfn_model, lr_policy, lr_flow, lr_back, back_model=False, conditional_flow_model=False, use_weight_decay=False, weight_decay=1e-7, energy=None):
    param_groups = [ {'params': gfn_model.t_model.parameters()},
                     {'params': gfn_model.s_model.parameters()},
                     {'params': gfn_model.joint_model.parameters()},
                     {'params': gfn_model.langevin_scaling_model.parameters()} ]
    if conditional_flow_model or energy=='vae':
        param_groups += [ {'params': gfn_model.flow_model.parameters(), 'lr': lr_flow} ]
    else:
        param_groups += [ {'params': [gfn_model.flow_model], 'lr': lr_flow} ]

    if back_model:
        param_groups += [ {'params': gfn_model.back_model.parameters(), 'lr': lr_back} ]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)
    return gfn_optimizer


def get_gfn_forward_loss(mode, init_state, gfn_model, log_reward, coeff_matrix, discretizer, exploration_std=None, return_exp=False, condition=None, repeats=10):
    if mode == 'tb':
        loss = fwd_tb(init_state, gfn_model, log_reward, discretizer, exploration_std, return_exp=return_exp, condition=condition)
    elif mode == 'tb-avg':
        loss = fwd_tb_avg(init_state, gfn_model, log_reward, discretizer, exploration_std, return_exp=return_exp, condition=condition)
    elif mode == 'cond-tb-avg':
        loss = fwd_tb_avg_cond(init_state, gfn_model, log_reward, discretizer, exploration_std, return_exp=return_exp, condition=condition, repeats=repeats)
    elif mode == 'db':
        loss = db(init_state, gfn_model, log_reward, discretizer, exploration_std, condition=condition)
    elif mode == 'subtb':
        loss = subtb(init_state, gfn_model, log_reward, coeff_matrix, discretizer, exploration_std, condition=condition)
    elif mode == 'pis':
        loss = pis(init_state, gfn_model, log_reward, discretizer, exploration_std=exploration_std, condition=condition)
    return loss


def get_gfn_backward_loss(mode, samples, gfn_model, log_reward, discretizer, exploration_std=None, condition=None, repeats=10):
    if mode == 'tb':
        loss = bwd_tb(samples, gfn_model, log_reward, discretizer, exploration_std, condition=condition)
    elif mode == 'tb-avg':
        loss = bwd_tb_avg(samples, gfn_model, log_reward, discretizer, exploration_std, condition=condition)
    elif mode == 'cond-tb-avg':
        loss = bwd_tb_avg_cond(samples, gfn_model, log_reward, discretizer, exploration_std, condition=condition, repeats=repeats)
    elif mode == 'mle':
        loss = bwd_mle(samples, gfn_model, log_reward, discretizer, exploration_std, condition=condition)
    return loss


def get_exploration_std(iter, exploratory, epochs, exploration_factor=0.1, exploration_wd=False):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1. - (iter / (epochs/2)))
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl


def uniform_discretizer(bsz, trajectory_length):
    return torch.linspace(0, 1, trajectory_length + 1).repeat(bsz, 1)


def random_discretizer(bsz, trajectory_length, max_ratio):
    x = (torch.rand(bsz, trajectory_length) * (max_ratio - 1) + 1).cumsum(1)
    x = torch.cat([torch.zeros(bsz, 1), x], 1) / x[:, -1].unsqueeze(1)
    return x


# def low_discrepancy_discretizer(bsz, traj_length=2):
#     u = torch.rand(1, traj_length).cumsum(1)
#     shift_vector = (torch.arange(bsz) / bsz).unsqueeze(1).repeat(1, traj_length-1)
#     u = (u/u[:, -1])[:, :-1]
#     timestep = u + shift_vector
#     timesteps_in_range = timestep % 1.0
#     timesteps_sorted, indices = torch.sort(timesteps_in_range, dim=-1, descending=False)
#     x = torch.cat([torch.zeros(bsz, 1), timesteps_sorted, torch.ones(bsz, 1)], dim=1)
#     return x


def low_discrepancy_discretizer(bsz, traj_length=2):
    u = torch.rand(1, traj_length-1)
    u_sorted, _ = torch.sort(u, dim=-1, descending=False)
    # print(u_sorted)
    # print(u_sorted.shape)
    shift_vector = (torch.arange(bsz) / bsz).unsqueeze(1).repeat(1, traj_length - 1)
    timestep = u + shift_vector
    timesteps_in_range = timestep % 1.0
    timesteps_sorted, indices = torch.sort(timesteps_in_range, dim=-1, descending=False)
    x = torch.cat([torch.zeros(bsz, 1), timesteps_sorted, torch.ones(bsz, 1)], dim=1)
    return x

    # old code below:
    # u = torch.rand(1)
    # shift_vector = torch.arange(bsz)/bsz
    # timestep = u + shift_vector
    # timestep_in_range = timestep % 1.0
    # timestep_in_range = timestep_in_range.unsqueeze(-1)
    # x = torch.cat([torch.zeros(bsz, 1), timestep_in_range, torch.ones(bsz, 1)], 1)
    # return x


def low_discrepancy_discretizer2(bsz, traj_length=2):
    s = traj_length - 1
    u = torch.rand(1, s)
    shift_vector = torch.arange(bsz) / bsz
    timestep = u + shift_vector.unsqueeze(-1)
    timestep_in_range = timestep % 1.0
    x = (timestep_in_range + torch.arange(s).unsqueeze(0)) / s
    x = torch.stack([col[torch.randperm(col.size(0))] for col in x.t()]).t()
    return x


def shifted_equidistant(bsz, traj_length, eps=1e-4):
    bound = 1 / traj_length - eps
    noise = torch.empty(bsz, 1).uniform_(- bound, bound)
    steps = (torch.arange(1, traj_length) / traj_length).unsqueeze(0) + noise
    return torch.cat([torch.zeros(bsz, 1), steps, torch.ones(bsz, 1)], dim=1)


def get_name(args):
    name = f'{args.discretizer}_{args.discretizer_traj_length}_steps_discretizer/'
    if args.discretizer == 'random':
        name += f'max_ratio_{args.discretizer_max_ratio}/'
    if args.traj_length_strategy == 'dynamic':
        name += f'dynamic_{args.min_traj_length}_{args.max_traj_length}/'

    if args.langevin:
        name += f'langevin_'
        if args.langevin_scaling_per_dimension:
            name += f'langevin_scaling_per_dimension_'
    if args.exploratory and (args.exploration_factor is not None):
        if args.exploration_wd:
            name += f'exploration_wd_{args.exploration_factor}_{name}_'
        else:
            name += f'exploration_{args.exploration_factor}_{name}_'

    if args.learn_pb:
        name = f'{name}learn_pb_scale_range_{args.pb_scale_range}_'

    if args.clipping:
        name = f'{name}clipping_lgv_{args.lgv_clip}_gfn_{args.gfn_clip}_'

    if args.mode_fwd == 'subtb':
        mode_fwd = f'subtb_subtb_lambda_{args.subtb_lambda}'
        if args.partial_energy:
            mode_fwd = f'{mode_fwd}_{args.partial_energy}'
    else:
        mode_fwd = args.mode_fwd

    if args.both_ways:
        ways = f'fwd_bwd/fwd_{mode_fwd}_bwd_{args.mode_bwd}'
    elif args.bwd:
        ways = f'bwd/bwd_{args.mode_bwd}'
    else:
        ways = f'fwd/fwd_{mode_fwd}'

    if args.local_search:
        local_search = f'local_search_iter_{args.max_iter_ls}_burn_{args.burn_in}_cycle_{args.ls_cycle}_step_{args.ld_step}_beta_{args.beta}_rankw_{args.rank_weight}_prioritized_{args.prioritized}'
        ways = f'{ways}/{local_search}'

    if args.pis_architectures:
        results = 'results_pis_architectures'
    else:
        results = 'results'

    name = f'{results}/{args.energy}/{name}gfn/{ways}/T_{args.T}/tscale_{args.t_scale}/lvr_{args.log_var_range}/'

    name = f'{name}/seed_{args.seed}/'

    return name
