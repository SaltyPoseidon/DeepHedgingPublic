# risk_rollout_buffer.py
"""
RiskRolloutBuffer
─────────────────
• хранит z_t = exp(-λ·V_t) вместо V_t;
• считает таргет по формуле
      z_t = exp(-λ r_t) · z_{t+1}^γ         (если шаг не терминальный)
  и   z_t = 1                               (если терминальный);
• advantages = z_target − z_pred            (Z-пространство);
• полностью векторизировано по n_envs.

Использование:
    from risk_rollout_buffer import RiskRolloutBuffer

    model = PPO(
        policy=RiskAwarePolicy,
        env=vec_env,
        n_steps=2048,
        rollout_buffer_class=RiskRolloutBuffer,
        # остальные гиперпараметры…
    )
"""
from typing import Optional, Generator, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import (
    RolloutBuffer,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

# --- фиксированная λ (single-lambda сценарий) ------------------------
LAMBDA_FIXED: float = 1


class RiskRolloutBuffer(RolloutBuffer):
    """
    Rollout-буфер с риск-аверсным TD для exp-ценности (Z-критика).
    Инженерно отличается только методом compute_returns_and_advantage.
    """

    def reset(self) -> None:  # => наследуем поведение SB3
        super().reset()

    # -----------------------------------------------------------------
    def compute_returns_and_advantage(  # noqa: D401
        self,
        last_values: th.Tensor,         # <- z_{T} для каждого env
        dones: np.ndarray,              # <- терминальность последнего шага
    ) -> None:
        """
        Заполняет self.returns (z-таргет) и self.advantages (z_tar − z_pred).

        :param last_values: Z-оценка последнего состояния (shape: n_envs)
        :param dones: терминальные ли последние шаги (shape: n_envs)
        """
        λ = LAMBDA_FIXED
        gamma = self.gamma

        # в numpy для скорости
        z_next = last_values.clone().cpu().numpy().flatten()        # shape (n_envs,)

        for step in reversed(range(self.buffer_size)):
            r_t = self.rewards[step]                                # shape (n_envs,)
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
            # Z-таргет: z_t = e^{-λ r_t} * z_{t+1}^γ   (если жив)
            z_target = (
                np.exp(-λ * r_t) * (z_next ** gamma) * next_non_terminal
                + (1.0 - next_non_terminal)         # если терминал → z_t = 1
            )

            self.returns[step] = z_target
            z_next = z_target                       # для следующей итерации

        # advantages = z_target − z_pred  (оба храним в self.returns / self.values)
        self.advantages = self.returns - self.values
        print(self.advantages.std(), self.values[:3], self.returns[:3])

    # -----------------------------------------------------------------
    # остальное оставляем без изменений → переиспользуем реализацию SB3
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,      # ← сюда кладём z_pred из критика
        log_prob: th.Tensor,
    ) -> None:
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        return super().get(batch_size)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        return super()._get_samples(batch_inds, env)
