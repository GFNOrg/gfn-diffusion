import torch
import numpy as np

def adjust_ld_step(current_ld_step, current_acceptance_rate, target_acceptance_rate=0.574, adjustment_factor=0.01):
    """
    Adjust the Langevin dynamics step size based on the current acceptance rate.
    
    :param current_ld_step: Current Langevin dynamics step size.
    :param current_acceptance_rate: Current observed acceptance rate.
    :param target_acceptance_rate: Target acceptance rate, default is 0.574.
    :param adjustment_factor: Factor to adjust the ld_step.
    :return: Adjusted Langevin dynamics step size.
    """
    if current_acceptance_rate > target_acceptance_rate:
        return current_ld_step + adjustment_factor * current_ld_step
    else:
        return current_ld_step - adjustment_factor * current_ld_step

def langevin_dynamics(x, log_reward, device, args):
    accepted_samples = []
    accepted_logr = []
    acceptance_rate_lst = []
    log_r_original = log_reward(x)
    acceptance_count = 0
    acceptance_rate = 0
    total_proposals = 0

    for i in range(args.max_iter_ls):
        x = x.requires_grad_(True)
        
        r_grad_original = torch.autograd.grad(log_reward(x).sum(), x)[0]
        if args.ld_schedule:
            ld_step = args.ld_step if i == 0 else adjust_ld_step(ld_step, acceptance_rate, target_acceptance_rate=args.target_acceptance_rate)
        else:
            ld_step = args.ld_step

        new_x = x + ld_step * r_grad_original.detach() + np.sqrt(2 * ld_step) * torch.randn_like(x, device=device)
        log_r_new = log_reward(new_x)
        r_grad_new = torch.autograd.grad(log_reward(new_x).sum(), new_x)[0]

        log_q_fwd = -(torch.norm(new_x - x - ld_step * r_grad_original, p=2, dim=1) ** 2) / (4 * ld_step)
        log_q_bck = -(torch.norm(x - new_x - ld_step * r_grad_new, p=2, dim=1) ** 2) / (4 * ld_step)

        log_accept = (log_r_new - log_r_original) + log_q_bck - log_q_fwd
        accept_mask = torch.rand(x.shape[0], device=device) < torch.exp(torch.clamp(log_accept, max=0))
        acceptance_count += accept_mask.sum().item()
        total_proposals += x.shape[0]

        x = x.detach()
        # After burn-in process
        if i > args.burn_in:
            accepted_samples.append(new_x[accept_mask])
            accepted_logr.append(log_r_new[accept_mask])
        x[accept_mask] = new_x[accept_mask]
        log_r_original[accept_mask] = log_r_new[accept_mask]

        if i % 5 == 0:
            acceptance_rate = acceptance_count / total_proposals
            if i>args.burn_in:
                acceptance_rate_lst.append(acceptance_rate)
            acceptance_count = 0
            total_proposals = 0

    return torch.cat(accepted_samples, dim=0), torch.cat(accepted_logr, dim=0)