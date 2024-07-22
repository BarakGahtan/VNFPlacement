import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=None, *args, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        if net_arch is None:
            net_arch = [256, 256]
        if activation_fn is None:
            activation_fn = nn.ReLU

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        # Define the shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], self.net_arch[0]),
            self.activation_fn(),
            nn.Linear(self.net_arch[0], self.net_arch[1]),
            self.activation_fn(),
        )

        # Define the action heads for each part of the MultiDiscrete action space
        self.action_heads = nn.ModuleList([nn.Linear(self.net_arch[-1], action_space.nvec[i]) for i in range(len(action_space.nvec))])

        # Define the value head
        self.value_head = nn.Linear(self.net_arch[-1], 1)

        # Distribution for actions
        self.distributions = [CategoricalDistribution(action_space.nvec[i]) for i in range(len(action_space.nvec))]

    def forward_features(self, obs):
        features = self.feature_extractor(obs)

        action_logits = [action_head(features) for action_head in self.action_heads]
        values = self.value_head(features)

        return action_logits, values

    def _get_action_dist_from_logits(self, action_logits):
        return [dist.proba_distribution(logits) for dist, logits in zip(self.distributions, action_logits)]

    def forward(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device)
        action_logits, values = self.forward_features(obs)
        action_distributions = self._get_action_dist_from_logits(action_logits)

        actions = [dist.get_actions(deterministic=deterministic) for dist in action_distributions]
        log_prob = [dist.log_prob(action) for dist, action in zip(action_distributions, actions)]

        return torch.cat(actions, dim=-1), values, torch.cat(log_prob, dim=-1)

    def _predict(self, observation, deterministic=False):
        action_logits, values = self.forward_features(observation)
        action_distributions = self._get_action_dist_from_logits(action_logits)

        actions = [dist.get_actions(deterministic=deterministic) for dist in action_distributions]

        return torch.cat(actions, dim=-1), values

    def evaluate_actions(self, obs, actions):
        action_logits, values = self.forward_features(obs)
        action_distributions = self._get_action_dist_from_logits(action_logits)

        action_log_probs = [dist.log_prob(action) for dist, action in zip(action_distributions, torch.split(actions, 1, dim=-1))]
        entropy = sum([dist.entropy() for dist in action_distributions])

        return values, torch.cat(action_log_probs, dim=-1), entropy

    def get_distribution(self, obs):
        action_logits, _ = self.forward_features(obs)
        action_distributions = self._get_action_dist_from_logits(action_logits)
        return action_distributions
