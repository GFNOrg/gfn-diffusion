import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .architectures import *
from utils import gaussian_params

logtwopi = math.log(2 * math.pi)


class GFN(nn.Module):
    def __init__(self, dim: int, s_emb_dim: int, hidden_dim: int,
                 harmonics_dim: int, t_dim: int, log_var_range: float = 4.,
                 t_scale: float = 1., langevin: bool = False, learned_variance: bool = True,
                 trajectory_length: int = 100, partial_energy: bool = False,
                 clipping: bool = False, lgv_clip: float = 1e4, gfn_clip: float = 1e4, pb_scale_range: float = 1.,
                 langevin_scaling_per_dimension: bool = True, conditional_flow_model: bool = False,
                 learn_pb: bool = False,
                 pis_architectures: bool = False, lgv_layers: int = 3, joint_layers: int = 2,
                 zero_init: bool = False, device=torch.device('cuda'), energy: str = None):
        super(GFN, self).__init__()
        self.dim = dim
        self.harmonics_dim = harmonics_dim
        self.t_dim = t_dim
        self.s_emb_dim = s_emb_dim

        self.trajectory_length = trajectory_length
        self.langevin = langevin
        self.learned_variance = learned_variance
        self.partial_energy = partial_energy
        self.t_scale = t_scale

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.langevin_scaling_per_dimension = langevin_scaling_per_dimension
        self.conditional_flow_model = conditional_flow_model
        self.learn_pb = learn_pb

        self.pis_architectures = pis_architectures
        self.lgv_layers = lgv_layers
        self.joint_layers = joint_layers

        self.pf_std_per_traj = np.sqrt(self.t_scale)
        self.dt = 1. / trajectory_length
        self.log_var_range = log_var_range
        self.device = device

        self.vae = True if energy == 'vae' else False

        self.t_model = TimeEncodingVAE(harmonics_dim, t_dim, hidden_dim)
        self.s_model = StateEncodingVAE(dim, 784, hidden_dim, s_emb_dim, num_layers=2)
        self.joint_model = JointPolicyVAE(dim, s_emb_dim, t_dim, hidden_dim, 2 * dim, joint_layers, zero_init)
        if learn_pb:
            self.back_model = JointPolicyVAE(dim, s_emb_dim, t_dim, hidden_dim, 2 * dim, joint_layers, zero_init)
        self.pb_scale_range = pb_scale_range

        if self.partial_energy:
            self.flow_model = FlowModelPartialEnergyVAE(dim, s_emb_dim, t_dim, hidden_dim, 1, joint_layers)
        else:
            self.flow_model = FlowModelVAE(dim, s_emb_dim, t_dim, hidden_dim, 1, joint_layers)

        if self.langevin_scaling_per_dimension:
            self.langevin_scaling_model = LangevinScalingModelVAE(s_emb_dim, t_dim, hidden_dim, dim,
                                                                  lgv_layers, zero_init)
        else:
            self.langevin_scaling_model = LangevinScalingModelVAE(s_emb_dim, t_dim, hidden_dim, 1,
                                                                  lgv_layers, zero_init)

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.

    def predict_next_state(self, s, t, log_r, condition=None):
        if self.langevin:
            s.requires_grad_(True)
            with torch.enable_grad():
                if condition is not None:
                    grad_log_r = torch.autograd.grad(log_r(s, condition).sum(), s)[0].detach()
                else:
                    grad_log_r = torch.autograd.grad(log_r(s).sum(), s)[0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.clipping:
                    grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

        t_lgv = t
        bsz = s.shape[0]
        t = self.t_model(t).repeat(bsz, 1)
        if condition is not None:
            s = self.s_model(s, condition)
        else:
            s = self.s_model(s)
        s_new = self.joint_model(s, t)

        if condition is not None:
            if self.partial_energy:
                flow = self.flow_model(s, t).squeeze(-1)
            else:
                flow = self.flow_model(s).squeeze(-1)
        else:
            flow = self.flow_model(s, t).squeeze(-1) if self.conditional_flow_model else self.flow_model

        if self.langevin:
            if self.pis_architectures or condition is not None:
                scale = self.langevin_scaling_model(t_lgv)
            else:
                scale = self.langevin_scaling_model(s, t)
            s_new[..., :self.dim] += scale * grad_log_r

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new, flow.squeeze(-1)

    def get_trajectory_fwd(self, s, exploration_std, log_r, condition=None):
        bsz = s.shape[0]

        logpf = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logpb = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logf = torch.zeros((bsz, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((bsz, self.trajectory_length + 1, self.dim), device=self.device)

        for i in range(self.trajectory_length):
            pfs, flow = self.predict_next_state(s, i * self.dt, log_r, condition)
            pf_mean, pflogvars = self.split_params(pfs)

            logf[:, i] = flow
            if self.partial_energy:
                ref_log_var = np.log(self.t_scale * max(1, i) * self.dt)
                log_p_ref = -0.5 * (logtwopi + ref_log_var + np.exp(-ref_log_var) * (s ** 2)).sum(1)
                logf[:, i] += (1 - i * self.dt) * log_p_ref + i * self.dt * log_r(s, condition)

            if exploration_std is None:
                pflogvars_sample = pflogvars.detach()
            else:
                expl = exploration_std(i)
                if expl <= 0.0:
                    pflogvars_sample = pflogvars.detach()
                else:
                    add_log_var = torch.full_like(pflogvars, np.log(exploration_std(i) / np.sqrt(self.dt)) * 2)
                    pflogvars_sample = torch.logaddexp(pflogvars, add_log_var).detach()

            s_ = s + self.dt * pf_mean.detach() + np.sqrt(self.dt) * (
                    pflogvars_sample / 2).exp() * torch.randn_like(s, device=self.device)

            noise = ((s_ - s) - self.dt * pf_mean) / (np.sqrt(self.dt) * (pflogvars / 2).exp())
            logpf[:, i] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(1)

            if self.learn_pb:
                t = self.t_model((i + 1) * self.dt).repeat(bsz, 1)
                if condition is not None:
                    pbs = self.back_model(self.s_model(s_, condition), t)
                else:
                    pbs = self.back_model(self.s_model(s_), t)
                dmean, dvar = gaussian_params(pbs)
                back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
            else:
                back_mean_correction, back_var_correction = torch.ones_like(s_), torch.ones_like(s_)

            if i > 0:
                back_mean = s_ - self.dt * s_ / ((i + 1) * self.dt) * back_mean_correction
                back_var = (self.pf_std_per_traj ** 2) * self.dt * i / (i + 1) * back_var_correction
                noise_backward = (s - back_mean) / back_var.sqrt()
                logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)

            s = s_
            states[:, i + 1] = s

        return states, logpf, logpb, logf

    def get_trajectory_bwd(self, s, exploration_std, log_r, condition=None):
        bsz = s.shape[0]
        logpf = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logpb = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logf = torch.zeros((bsz, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((bsz, self.trajectory_length + 1, self.dim), device=self.device)
        states[:, -1] = s

        for i in range(self.trajectory_length):
            if i < self.trajectory_length - 1:
                if self.learn_pb:
                    t = self.t_model(1. - i * self.dt).repeat(bsz, 1)
                    if condition is not None:
                        pbs = self.back_model(self.s_model(s, condition), t)
                    else:
                        pbs = self.back_model(self.s_model(s), t)
                    dmean, dvar = gaussian_params(pbs)
                    back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                    back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
                else:
                    back_mean_correction, back_var_correction = torch.ones_like(s), torch.ones_like(s)

                mean = s - self.dt * s / (1. - i * self.dt) * back_mean_correction
                var = ((self.pf_std_per_traj ** 2) * self.dt * (1. - (i + 1) * self.dt)) / (
                            1 - i * self.dt) * back_var_correction
                s_ = mean.detach() + var.sqrt().detach() * torch.randn_like(s, device=self.device)
                noise_backward = (s_ - mean) / var.sqrt()
                logpb[:, self.trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + var.log()).sum(1)
            else:
                s_ = torch.zeros_like(s)

            pfs, flow = self.predict_next_state(s_, (1. - (i + 1) * self.dt), log_r, condition)
            pf_mean, pflogvars = self.split_params(pfs)

            logf[:, self.trajectory_length - i - 1] = flow
            if self.partial_energy:
                ref_log_var = np.log(self.t_scale * max(1, self.trajectory_length - i - 1) * self.dt)
                log_p_ref = -0.5 * (logtwopi + ref_log_var + np.exp(-ref_log_var) * (s ** 2)).sum(1)
                logf[:, self.trajectory_length - i - 1] += (i + 1) * self.dt * log_p_ref + (
                        self.trajectory_length - i - 1) * self.dt * log_r(s, condition)

            noise = ((s - s_) - self.dt * pf_mean) / (np.sqrt(self.dt) * (pflogvars / 2).exp())
            logpf[:, self.trajectory_length - i - 1] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(
                1)

            s = s_
            states[:, self.trajectory_length - i - 1] = s

        return states, logpf, logpb, logf

    def sample(self, batch_size, log_r, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, None, log_r, condition)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, exploration_std, log_r=None, condition=condition)[0][:, -1]

    def forward(self, s, exploration_std=None, log_r=None, condition=None):
        return self.get_trajectory_fwd(s, exploration_std, log_r, condition)
