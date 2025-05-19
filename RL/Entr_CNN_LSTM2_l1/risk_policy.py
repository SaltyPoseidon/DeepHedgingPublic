# risk_policy.py
"""
RiskAwarePolicy  –  Actor-Critic в V-шкале
------------------------------------------
 • critic прогнозирует V(s) = −log z / λ   (scalar);
 • actor работает как обычно;
 • подходит к RiskRolloutBufferV.
"""

from __future__ import annotations
from functools import partial
from typing import Any, Dict, Optional, Union

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import (
    ActorCriticPolicy,
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

# ────────────────────────────────────────────────────────────────
LAMBDA_FIXED: float = 1   # здесь λ только для helper-функции
# ────────────────────────────────────────────────────────────────


class RiskAwarePolicy(ActorCriticPolicy):
    """Кастом-policy для Deep Hedging: critic → V-output."""

    # ---------- BUILD --------------------------------------------------
    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_pi = self.mlp_extractor.latent_dim_pi

        # actor-голова (как в SB3)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_pi, latent_sde_dim=latent_pi, log_std_init=self.log_std_init
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_pi)
        else:
            raise NotImplementedError

        # critic: обычная linear без sigmoid
        self.v_head = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # → ортогональная инициализация (опц.)
        if self.ortho_init:
            gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.v_head: 1.0,
            }
            if isinstance(self.features_extractor, FlattenExtractor):
                gains[self.features_extractor] = np.sqrt(2)
            for module, g in gains.items():
                module.apply(partial(self.init_weights, gain=g))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ---------- FORWARD ------------------------------------------------
    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            lat_pi, lat_vf = self.mlp_extractor(features)
        else:
            pi_f, vf_f = features
            lat_pi = self.mlp_extractor.forward_actor(pi_f)
            lat_vf = self.mlp_extractor.forward_critic(vf_f)

        v_pred = self.v_head(lat_vf).squeeze(-1)        # (batch,)

        dist = self._get_action_dist_from_latent(lat_pi)
        actions = dist.get_actions(deterministic=deterministic)
        logp = dist.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, v_pred, logp

    # ---------- PREDICT / EVAL ----------------------------------------
    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        features = super().extract_features(obs, self.vf_features_extractor)
        lat_vf = self.mlp_extractor.forward_critic(features)
        return self.v_head(lat_vf).squeeze(-1)

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            lat_pi, lat_vf = self.mlp_extractor(features)
        else:
            pi_f, vf_f = features
            lat_pi = self.mlp_extractor.forward_actor(pi_f)
            lat_vf = self.mlp_extractor.forward_critic(vf_f)

        dist = self._get_action_dist_from_latent(lat_pi)
        logp = dist.log_prob(actions)
        v_pred = self.v_head(lat_vf).squeeze(-1)
        entropy = dist.entropy()
        return v_pred, logp, entropy
