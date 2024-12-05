"""
Custom Concave-Augmented Pareto Q-Learning (CAPQL)
for implementation in UAV-IoT systems

Author: Mason Conkel
Creation Date: 11/29/2024
Last Update: 11/29/2024
"""
import copy

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Adam

# Other imports
import copy
import os
import math
import random
import numpy as np
from typing import Any, Tuple

# Put during model creation
from hydra.core.config_store import ConfigStore

# Global Variables
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def update_Q(target, source, tau) -> None:
    """
    Objective:
        Function for updating Target Network parameters using Source Network

    Inputs:
        :param target:  -> Target Q-Network
        :param source:  -> Source Q-Network
        :param tau:     -> Weight of each network parameter for target update.
    """
    for target_param, param in zip(target.paramenters(), source.paramaeters()):
        target_param.copy_(target_param.data * (1 - tau) + param.data * tau)


def weights_init_(m) -> None:
    """
    Objective:
        Function for the creation of weights matrix

    Inputs:
        :param m:   -> Internal parameter from torch.nn calls during model creation
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Weight_Sampler_Angle(object):
    def __init__(self, rwd_dim, angle, w=None):
        self.rwd_dim = rwd_dim
        self.angle = angle
        if w is None:
            w = torch.ones(rwd_dim)
        w = w / torch.norm(w)
        self.w = w

    def sample(self, n_sample):
        s = torch.normal(torch.zeros(n_sample, self.rwd_dim))

        # remove fluctuation on dir w
        s = s - (s @ self.w).view(-1, 1) * self.w.view(1, -1)

        # normalize it
        s = s / torch.norm(s, dim=1, keepdim=True)

        # sample angle
        s_angle = torch.rand(n_sample, 1) * self.angle

        # compute shifted vector from w
        w_sample = torch.tan(s_angle) * s + self.w.view(1, -1)

        w_sample = w_sample / torch.norm(w_sample, dim=1, keepdim=True, p=1)

        return w_sample


class Weight_Sampler_Pos:
    def __init__(self, rwd_dim):
        self.rwd_dim = rwd_dim

    def sample(self, n_sample):
        # sample from sphrical normal distribution
        s = torch.normal(torch.zeros(n_sample, self.rwd_dim))

        # flip all negative weights to be non-negative
        s = torch.abs(s)

        # normalize
        s = s / torch.norm(s, dim=1, keepdim=True, p=1)

        return s


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, weights, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, weights, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, w, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, w, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, rwd_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions + rwd_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rwd_dim)
        )

        # Q2 architecture
        self.Q2 = copy.deepcopy(self.Q1)
        self.rwd_dim = rwd_dim
        self.apply(weights_init_)

    def forward(self, state, action, w, h_op=False):
        if not h_op:
            # 1) Prepare input
            # xu: batch_size x (state_dim + action_dim + rwd_dim)
            xu = torch.cat([state, action, w], 1)

            # 2) Evaluate
            # x1, x2: batch_size x rwd_dim
            x1 = self.Q1(xu)
            x2 = self.Q2(xu)
            return x1, x2
        else:

            # For qenv_ctn H operation

            with torch.no_grad():
                batch_size = n_weight = len(state)

                # 1) Prepare input
                # xu: batch_size x (state_dim + action_dim)
                state_action = torch.cat([state, action], 1)

                # state_action: batch_size x (state_dim + action_dim) -> batch_size x 1 x (state_dim + action_dim)
                # w: n_prob_weight x rwd_dim -> 1 x n_prob_weight x rwd_dim
                # Concatenate to get: batch_size x n_prob_weight x (state_dim + action_dim + rwd_dim)

                xu_expand = torch.cat(
                    (
                        state_action.unsqueeze(1).expand(-1, n_weight, -1),
                        w.unsqueeze(0).expand(batch_size, -1, -1),
                    ),
                    dim=-1
                )

                # 2) Evaluate to get Q values
                # q1_expand, q2_expand: batch_size x n_prob_weight x rwd_dim
                q1_expand = self.Q1(xu_expand)
                q2_expand = self.Q2(xu_expand)
                q_expand = torch.stack([q1_expand, q2_expand], 2).view(batch_size, n_weight * 2, self.rwd_dim)

                # 3) Compute projection
                # w: batch_size x rwd_dim
                # q1_expand, q2_expand: batch_size x n_prob_weight x rwd_dim
                # proj_1, proj_2: batch_size x n_prob_weight
                proj_1 = (w.unsqueeze(1) * q1_expand).sum(-1)
                proj_2 = (w.unsqueeze(1) * q2_expand).sum(-1)

                # max_proj_1, max_proj_2: batch_size
                # max_id_1, max_id_2: batch_size
                max_proj_1, max_id_1 = torch.max(proj_1, dim=1)
                max_proj_2, max_id_2 = torch.max(proj_2, dim=1)

                # find the network gives the smaller projection
                # first_net_smaller_mask: batch_size
                first_net_smaller_mask = (max_proj_1 < max_proj_2).int().unsqueeze(-1)

            # compute again for the max the projection with gradient recorded
            q1_max = self.Q1(torch.cat([state, action, w[max_id_1]], 1))
            q2_max = self.Q2(torch.cat([state, action, w[max_id_2]], 1))
            q = q1_max * first_net_smaller_mask + q2_max * (1 - first_net_smaller_mask)

            return q


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, rwd_dim, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs + rwd_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, w):

        state_comp = torch.cat((state, w), dim=1)

        x = func.relu(self.linear1(state_comp))
        x = func.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state, w):
        # for each state in the mini-batch, get its mean and std
        mean, log_std = self.forward(state, w)
        std = log_std.exp()

        # sample actions
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))

        # restrict the outputs
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # compute the prob density of the samples

        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        # compute the log_prob as the normal distribution sample is processed by tanh
        #       (re-parameterization trick)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        log_prob = log_prob.clamp(-1e3, 1e3)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class capql_agent(object):
    """
    Concave-Augmented Pareto Q-Learning

    Needs:
        Define action space such that action_space.shape[0] represents a gym-wrapped action space
            - Action space with action_space = None may have only the number of actions as the space itself.
            - Action space == num_actions -> pass None to Gaussian Network and num actions to all networks.

        Rwd_dim
            - Reward functions can be represented by an array instead of a singular value

        Check and adjust uav_config for configuration of hyperparameters
    """
    def __init__(self, num_inputs: int, action_space: np.ndarray, rwd_dim: np.ndarray, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2, model_type: str = 'CAPQL', target_update_interval: int = 1,
                 cuda: bool = True, hidden_size: int = 256, lr: float = 0.0003):
        # Rip information from config file
        self.gamma = gamma
        self.tau = tau
        self. alpha = alpha

        self.policy_type = model_type
        self.target_update_interval = target_update_interval
        self.device = torch.device("cuda" if cuda else "cpu")

        # Define two Q Networks for critic
        self.critic = QNetwork(num_inputs, action_space.shape[0], rwd_dim,
                               hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        # Define two Q Networks for target
        self.critic_target = copy.deepcopy(self.critic)
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], rwd_dim, hidden_size).to(self.device)
        self.policy_optim = Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, w, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        w = torch.FloatTensor(w).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.policy.sample(state, w)
        else:
            _, _, action = self.policy.sample(state, w)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates, delta=0.1):
        # Sample a batch from memory
        state_batch, action_batch, w_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        w_batch = torch.FloatTensor(w_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        ########################
        # train the Q network  #
        ########################

        # compute next_q_value target
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, w_batch)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, w_batch,
                                                                  h_op=False)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # update
        qf1, qf2 = self.critic(state_batch, action_batch, w_batch)
        qf1_loss = func.mse_loss(qf1, next_q_value)
        qf2_loss = func.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        ############################
        # train the policy network #
        ############################

        # for each sample in minibatch, sample an action in 3-d space
        # pi is the action, log_pi is its log probability
        pi, log_pi, _ = self.policy.sample(state_batch, w_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi, w_batch, h_op=False)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # minimize KL divergence
        min_qf_pi = (min_qf_pi * w_batch).sum(dim=-1, keepdim=True)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # sync the Q networks
        if updates % self.target_update_interval == 0:
            update_Q(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()