from plot_utils import *
import argparse
import torch
import os
import numpy as np

from utils import (set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name, uniform_discretizer, random_discretizer,
                   low_discrepancy_discretizer, low_discrepancy_discretizer2, shifted_equidistant)
from buffer import ReplayBuffer
from langevin import langevin_dynamics
from models import GFN
from gflownet_losses import *
from energies import *
from evaluations import *

import matplotlib.pyplot as plt
from tqdm import trange

WANDB = True

if WANDB:
    import wandb

parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--buffer_size', type=int, default=300 * 100 * 2)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='vae',
                    choices=('vae'))
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', 'cond-tb-avg', 'pis'))
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle', 'cond-tb-avg'))
parser.add_argument('--both_ways', action='store_true', default=False)
parser.add_argument('--repeats', type=int, default=10)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=200)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=100)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

# langevin temperature
parser.add_argument('--ld_beta', type=float, default=5.)

# Langevin dynamics schedule
parser.add_argument('--ld_schedule', action='store_true', default=False)
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)
################################################################


# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################

# Stepwise scheduler
parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--step_point', type=int, default=7000)

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--discretizer', type=str, default="random",
                    choices=('random', 'uniform', 'low_discrepancy', 'low_discrepancy2', 'equidistant', 'adaptive'))
parser.add_argument('--discretizer_max_ratio', type=float, default=10.0)
parser.add_argument('--discretizer_traj_length', type=int, default=100)
parser.add_argument('--traj_length_strategy', type=str, default="static", choices=('static', 'dynamic'))
parser.add_argument('--min_traj_length', type=int, default=10)
parser.add_argument('--max_traj_length', type=int, default=100)
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 100
final_eval_data_size = 100
plot_data_size = 16
final_plot_data_size = 16

if args.pis_architectures:
    args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.discretizer_traj_length).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True


def get_energy():
    if args.energy == 'vae':
        energy = VAEEnergy(device=device, batch_size=args.batch_size)
    else:
        return NotImplementedError
    return energy


def plot_step(energy, gfn_model, name):
    if args.energy == 'vae':
        batch_size = plot_data_size
        real_data = energy.sample_evaluation_subset(batch_size)

        fig_real_data, ax_real_data = get_vae_images(real_data.detach().cpu())

        vae_samples_mu, vae_samples_logvar = energy.vae.encode(real_data)
        vae_z = energy.vae.reparameterize(vae_samples_mu, vae_samples_logvar)
        vae_samples = energy.vae.decode(vae_z)
        fig_vae_samples, ax_vae_samples = get_vae_images(vae_samples.detach().cpu())

        gfn_samples_z = gfn_model.sample(batch_size, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward, condition=real_data, pis=True if args.mode_fwd == 'pis' else False)
        gfn_samples = energy.vae.decode(gfn_samples_z)
        fig_gfn_samples, ax_gfn_samples = get_vae_images(gfn_samples.detach().cpu())

        fig_real_data.savefig(f'{name}real_data.pdf', bbox_inches='tight')
        fig_vae_samples.savefig(f'{name}vae_samples.pdf', bbox_inches='tight')
        fig_gfn_samples.savefig(f'{name}gfn_samples.pdf', bbox_inches='tight')

        return {"visualization/real_data": wandb.Image(fig_to_image(fig_real_data)),
                "visualization/vae_samples": wandb.Image(fig_to_image(fig_vae_samples)),
                "visualization/gfn_samples": wandb.Image(fig_to_image(fig_gfn_samples))}

    else:
        return {}

def plot_step_K_step_discretizer(energy, gfn_model, name):
    if args.energy == 'vae':
        batch_size = plot_data_size
        real_data = energy.sample_evaluation_subset(batch_size)

        fig_real_data, ax_real_data = get_vae_images(real_data.detach().cpu())

        vae_samples_mu, vae_samples_logvar = energy.vae.encode(real_data)
        vae_z = energy.vae.reparameterize(vae_samples_mu, vae_samples_logvar)
        vae_samples = energy.vae.decode(vae_z)
        fig_vae_samples, ax_vae_samples = get_vae_images(vae_samples.detach().cpu())

        gfn_samples_z = gfn_model.sample(batch_size, lambda bsz: uniform_discretizer(bsz, args.discretizer_traj_length), energy.log_reward, condition=real_data, pis=True if args.mode_fwd == 'pis' else False)
        gfn_samples = energy.vae.decode(gfn_samples_z)
        fig_gfn_samples, ax_gfn_samples = get_vae_images(gfn_samples.detach().cpu())

        fig_real_data.savefig(f'{name}{args.discretizer_traj_length}_steps_real_data.pdf', bbox_inches='tight')
        fig_vae_samples.savefig(f'{name}{args.discretizer_traj_length}_stepsvae_samples.pdf', bbox_inches='tight')
        fig_gfn_samples.savefig(f'{name}{args.discretizer_traj_length}_stepsgfn_samples.pdf', bbox_inches='tight')

        return {f"visualization_{args.discretizer_traj_length}_steps/real_data": wandb.Image(fig_to_image(fig_real_data)),
                f"visualization_{args.discretizer_traj_length}_steps/vae_samples": wandb.Image(fig_to_image(fig_vae_samples)),
                f"visualization_{args.discretizer_traj_length}_steps/gfn_samples": wandb.Image(fig_to_image(fig_gfn_samples))}

    else:
        return {}


def eval_step(eval_data, energy, gfn_model, final_eval=False, condition=None):
    gfn_model.eval()
    metrics = dict()
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics['final_eval/log_Z'], metrics['final_eval/log_Z_lb'], metrics[
            'final_eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward, condition=condition, pis=False)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
            'eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward, condition=condition, pis=False)
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    elif condition is not None:
        sample_based_metrics = None
    else:
        if final_eval:
            metrics['final_eval/mean_log_likelihood'] = mean_log_likelihood(eval_data, gfn_model,
                                                                            lambda bsz: uniform_discretizer(bsz, args.T),
                                                                            energy.log_reward,
                                                                            condition=condition)
        else:
            metrics['eval/mean_log_likelihood'] = mean_log_likelihood(eval_data, gfn_model,
                                                                      lambda bsz: uniform_discretizer(bsz, args.T),
                                                                      energy.log_reward,
                                                                      condition=condition)
        metrics.update(get_sample_metrics(samples, eval_data, final_eval))
    gfn_model.train()
    return metrics


def eval_step_K_step_discretizer(eval_data, energy, gfn_model, final_eval=False, condition=None):
    gfn_model.eval()
    metrics = dict()
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics[f'final_eval_{args.discretizer_traj_length}_steps/log_Z'], metrics[f'final_eval_{args.discretizer_traj_length}_steps/log_Z_lb'], metrics[
            f'final_eval_{args.discretizer_traj_length}_steps/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, lambda bsz: uniform_discretizer(bsz, args.discretizer_traj_length), energy.log_reward, condition=condition, pis=False)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics[f'eval_{args.discretizer_traj_length}_steps/log_Z'], metrics[f'eval_{args.discretizer_traj_length}_steps/log_Z_lb'], metrics[
            f'eval_{args.discretizer_traj_length}_steps/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, lambda bsz: uniform_discretizer(bsz, args.discretizer_traj_length), energy.log_reward, condition=condition, pis=False)
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    elif condition is not None:
        sample_based_metrics = None
    else:
        if final_eval:
            metrics[f'final_eval_{args.discretizer_traj_length}_steps/mean_log_likelihood'] = mean_log_likelihood(eval_data, gfn_model,
                                                                            lambda bsz: uniform_discretizer(bsz, args.discretizer_traj_length),
                                                                            energy.log_reward,
                                                                            condition=condition)
        else:
            metrics[f'eval_{args.discretizer_traj_length}_steps/mean_log_likelihood'] = mean_log_likelihood(eval_data, gfn_model,
                                                                      lambda bsz: uniform_discretizer(bsz, args.discretizer_traj_length),
                                                                      energy.log_reward,
                                                                      condition=condition)
        metrics.update(get_sample_metrics(samples, eval_data, final_eval))
    gfn_model.train()
    return metrics


def train_step(energy, gfn_model, gfn_optimizer, it, exploratory, epochs, buffer, buffer_ls, exploration_factor,
               exploration_wd, condition=None, repeats=10):
    gfn_model.zero_grad()

    traj_length = args.discretizer_traj_length if args.traj_length_strategy == 'static' \
        else np.random.randint(low=args.min_traj_length, high=args.max_traj_length + 1)
    if args.discretizer == 'random':
        discretizer = lambda bsz: random_discretizer(bsz, traj_length, max_ratio=args.discretizer_max_ratio)
    elif args.discretizer == 'low_discrepancy':
        discretizer = lambda bsz: low_discrepancy_discretizer(bsz, traj_length)
    elif args.discretizer == 'low_discrepancy2':
        discretizer = lambda bsz: low_discrepancy_discretizer2(bsz, traj_length)
    elif args.discretizer == 'equidistant':
        discretizer = lambda bsz: shifted_equidistant(bsz, traj_length)
    else:
        discretizer = lambda bsz: uniform_discretizer(bsz, traj_length)

    exploration_std = get_exploration_std(it, exploratory, epochs, exploration_factor, exploration_wd)
    if args.both_ways:
        if it % 2 == 0:
            if args.sampling == 'buffer':
                loss, states, _, _, log_r = fwd_train_step(energy, gfn_model, discretizer, exploration_std, return_exp=True,
                                                           condition=condition, repeats=repeats)

                states = states[:, -1]
                states = states.view(args.batch_size, repeats, -1)
                log_r = log_r.view(args.batch_size, repeats)
                states = states[torch.arange(args.batch_size), torch.argmax(log_r, dim=1)]

                log_r = log_r[torch.arange(args.batch_size), torch.argmax(log_r, dim=1)]

                buffer.add(states, log_r, condition=condition)
            else:
                loss = fwd_train_step(energy, gfn_model, discretizer, exploration_std, condition=condition, repeats=repeats)
        else:
            loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, discretizer, exploration_std, it=it, condition=condition,
                                  repeats=repeats)

    elif args.bwd:
        loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, discretizer, exploration_std, it=it, condition=condition,
                              repeats=repeats)
    else:
        loss = fwd_train_step(energy, gfn_model, discretizer, exploration_std, condition=condition, repeats=repeats)

    loss.backward()
    gfn_optimizer.step()
    return loss.item()


def fwd_train_step(energy, gfn_model, discretizer, exploration_std, return_exp=False, condition=None, repeats=10):
    init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)

    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, coeff_matrix, discretizer,
                                exploration_std=exploration_std, return_exp=return_exp, condition=condition,
                                repeats=repeats)
    return loss


def bwd_train_step(energy, gfn_model, buffer, buffer_ls, discretizer, exploration_std=None, it=0, condition=None, repeats=10):
    if args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(args.batch_size, exploration_std, condition=condition).to(device)
    elif args.sampling == 'energy':
        samples = energy.sample(args.batch_size).to(device)
    elif args.sampling == 'buffer':
        if args.local_search:
            if it % args.ls_cycle < 2:
                samples, _, condition, _ = buffer.sample()
                samples = samples.detach()
                condition = condition.detach()
                local_search_samples, log_r, condition = langevin_dynamics(samples, energy.log_reward, device, args,
                                                                           condition=condition)
                buffer_ls.add(local_search_samples.detach(), log_r.detach(), condition=condition)

            samples, log_r, condition, _ = buffer_ls.sample()

        else:

            samples, _, condition, _ = buffer.sample().detach()

    loss = get_gfn_backward_loss(args.mode_bwd, samples, gfn_model, energy.log_reward, discretizer,
                                 exploration_std=exploration_std, condition=condition, repeats=repeats)

    return loss


def train():
    name = get_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    energy = get_energy()
    if args.energy == 'vae':
        eval_data = energy.sample(eval_data_size, evaluation=True).to(device)
        final_eval_data = energy.sample(final_eval_data_size, evaluation=True).to(device)
    else:
        eval_data = energy.sample(eval_data_size).to(device)
        final_eval_data = energy.sample(
            final_eval_data_size).to(device)

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    if WANDB:
        wandb.init(project="GFN Conditional Energy", config=config, name=name)

    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device, energy=args.energy).to(
        device)



    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay,
                                      args.energy)

    if args.scheduler == True:
        lambda_function = lambda iteration: 0.1 if iteration >= args.step_point else 1.0

        # Initialize the scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(gfn_optimizer, lr_lambda=lambda_function)

    print(gfn_model)
    metrics = dict()

    buffer = ReplayBuffer(args.buffer_size, device, energy.log_reward,args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    buffer_ls = ReplayBuffer(args.buffer_size, device, energy.log_reward,args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                             rank_weight=args.rank_weight, prioritized=args.prioritized)
    gfn_model.train()
    for i in trange(args.epochs + 1):
        if args.energy == 'vae':
            condition = energy.sample(args.batch_size)
        else:
            condition = None

        metrics['train/loss'] = train_step(energy, gfn_model, gfn_optimizer, i, args.exploratory, args.epochs,
                                           buffer, buffer_ls, args.exploration_factor, args.exploration_wd,
                                           condition=condition, repeats=args.repeats)

        if args.scheduler == True:
            scheduler.step()
        if i % 1000 == 0:
            if args.energy == 'vae':
                condition = energy.sample(eval_data_size, evaluation=True)
            else:
                condition = None
            metrics.update(eval_step(eval_data, energy, gfn_model, final_eval=False, condition=condition))
            print('logz:', metrics['eval/log_Z_lb'])
            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics['eval/log_Z_learned']

            metrics.update(eval_step_K_step_discretizer(eval_data, energy, gfn_model, final_eval=False, condition=condition))
            print('logz:', metrics[f'eval_{args.discretizer_traj_length}_steps/log_Z_learned'])
            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics[f'eval_{args.discretizer_traj_length}_steps/log_Z_learned']
            if WANDB:
                images = plot_step(energy, gfn_model, name)
                metrics.update(images)
                plt.close('all')
                images_k_step = plot_step_K_step_discretizer(energy, gfn_model, name)
                metrics.update(images_k_step)
                plt.close('all')
                wandb.log(metrics, step=i)
            if i % 1000 == 0:
                torch.save(gfn_model.state_dict(), f'{name}model.pt')
                torch.save({
                    'epoch': i,
                    'model_state_dict': gfn_model.state_dict(),
                    'optimizer_state_dict': gfn_optimizer.state_dict(),
                    'loss': metrics['train/loss'],
                }, f'{name}model.pt')

    if args.energy == 'vae':
        condition = energy.sample(eval_data_size, evaluation=True)
    else:
        condition = None
    eval_results = eval_step(final_eval_data, energy, gfn_model, final_eval=True, condition=condition)
    metrics.update(eval_results)
    if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
        del metrics['final_eval/log_Z_learned']

    eval_results = eval_step_K_step_discretizer(final_eval_data, energy, gfn_model, final_eval=True, condition=condition)
    metrics.update(eval_results)
    if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
        del metrics[f'final_eval_{args.discretizer_traj_length}_steps/log_Z_learned']
    torch.save({
        'epoch': i,
        'model_state_dict': gfn_model.state_dict(),
        'optimizer_state_dict': gfn_optimizer.state_dict(),
        'loss': metrics['train/loss'],
    }, f'{name}model_final.pt')


def eval():
    pass


if __name__ == '__main__':
    if args.eval:
        eval()
    else:
        train()

