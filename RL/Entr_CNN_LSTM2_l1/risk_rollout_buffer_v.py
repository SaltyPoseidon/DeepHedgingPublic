# risk_rollout_buffer_v.py
"""
RiskRolloutBufferV – Rollout-буфер для экспоненциальной полезности в V-шкале.
Алгоритм:
    • в add() кладём V_pred;
    • при post-processing считаем
           z_t     = exp(-λ r_t) * z_{t+1}^γ,
           V_targ  = -log(z_t) / λ,
           adv     = V_targ - V_pred.
"""
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

LAMBDA_FIXED = 1  # такая же λ

class RiskRolloutBufferV(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )

    # ---------- главное изменение ------------------------------------
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        last_values = V_pred(s_{t+1})  (последний шаг каждой среды)
        """
        last_v_pred = last_values.cpu().numpy().flatten()
        last_z      = np.exp(-LAMBDA_FIXED * last_v_pred)  # z_{t+1}

        # пробегаем назад и считаем Z-таргет
        z_next = last_z
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                non_term = 1.0 - dones.astype(np.float32)
            else:
                non_term = 1.0 - self.episode_starts[step + 1]

            r_t   = self.rewards[step]
            z_t   = np.exp(-LAMBDA_FIXED * r_t) * (z_next ** (self.gamma * non_term))
            v_targ = -np.log(np.clip(z_t, 1e-8, 1.0)) / LAMBDA_FIXED

            self.returns[step]    = v_targ
            self.advantages[step] = v_targ - self.values[step]

            z_next = z_t  # переносим для следующей итерации

    # ---------- get() как в RolloutBuffer без изменений --------------
