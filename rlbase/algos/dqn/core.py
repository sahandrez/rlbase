import numpy as np

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        q = self.q(obs)
        return torch.squeeze(q, -1)     # Critical to ensure q has right shape.


class MLPQPolicy(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.Tanh):
        super().__init__()
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n

        # build the Q-network that returns action values of Q(s_t)
        self.q_net = MLPQFunction(self.obs_dim, self.act_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            q = self.q_net(obs)
            a = q.argmax()
        return a.numpy(), q.numpy()

    def act(self, obs):
        return self.step(obs)[0]
