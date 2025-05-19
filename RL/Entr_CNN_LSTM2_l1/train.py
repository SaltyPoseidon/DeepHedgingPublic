# train.py
"""
Обучает PPO-агента на среде ABMDeepHedgeEnv.
✓ CNN-кодировщик стакана (DeepLOBEncoder)
✓ Кастом-критик в V-шкале (RiskAwarePolicy)
✓ Экспоненциальная utility (RiskRolloutBufferV)

Запуск:
    python train.py
"""
from __future__ import annotations
import multiprocessing as mp

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from .abm_env              import ABMDeepHedgeEnv
from .lob_cnn              import DeepLOBEncoder        # CNN-extractor
from .risk_policy          import RiskAwarePolicy       # actor-critic (V-scale)
from .risk_rollout_buffer_v import RiskRolloutBufferV   # exp-utility buffer
from .util_callback        import UtilityCallback       # logs mean-utility
from .save_best_utility import SaveBestUtilityCallback

# ────────────────────────────────────  ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ  ──
N_ENVS        = 8
LAMBDA_FIXED  = 1              # λ в utility
TOTAL_STEPS   = 500_000           # n_envs × n_steps × updates
N_STEPS       = 4096              # кратно N_ENVS
BATCH_SIZE    = 2048

# ───────────────────────────────  фабрика окружений  ──────────
def make_env(rank: int, base_seed: int = 0):
    """Создаёт один Monitor-обёрнутый экземпляр среды."""
    def _init():
        env = ABMDeepHedgeEnv(lambda_fixed=LAMBDA_FIXED,
                              seed=base_seed + rank)
        return Monitor(env)
    return _init

# ───────────────────────────────────────────────  main()  ─────
def main() -> None:
    # 1. Векторная среда + нормализация наблюдений
    vec_env = SubprocVecEnv([make_env(i, base_seed=1234) for i in range(N_ENVS)])
    vec_env = VecNormalize(vec_env,
                           norm_obs=True,
                           norm_reward=False,
                           clip_obs=10.0)

    # 2. Отдельная eval-среда для callback (utility)
    eval_env   = ABMDeepHedgeEnv(lambda_fixed=LAMBDA_FIXED, seed=999)
    utility_cb = UtilityCallback(eval_env,
                                 n_eval_episodes=20,
                                 eval_freq=10_000)

    # 3. Параметры policy: CNN-extractor  →  MLP головы
    policy_kwargs = dict(
        features_extractor_class  = DeepLOBEncoder,
        features_extractor_kwargs = dict(lob_K=100, port_dim=16, out_dim=128),
        net_arch                  = [dict(pi=[256, 256, 256],
                                          vf=[256, 256, 256])],
        activation_fn             = nn.ELU,
        ortho_init                = False,      # Xavier по умолчанию
    )

    # 4. Создаём модель PPO с кастом-буфером
    model = PPO(
        policy               = RiskAwarePolicy,
        env                  = vec_env,
        rollout_buffer_class = RiskRolloutBufferV,
        n_steps              = N_STEPS,
        batch_size           = BATCH_SIZE,
        learning_rate        = 1e-4,
        gamma                = 0.98,
        gae_lambda           = 1.0,
        clip_range           = 0.3,
        clip_range_vf        = 10.0,
        ent_coef             = 1e-3,
        max_grad_norm        = 1.0,
        policy_kwargs        = policy_kwargs,
        verbose              = 1,
    )

    # 5. Обучение
    model.learn(total_timesteps=TOTAL_STEPS,
                callback=utility_cb)

    # 6. Сохраняем веса и нормализатор
    model.save("ppo_deephedge_cnn_l1")
    vec_env.save("vecnorm_cnn_l1.pkl")
    print("✔ Model and normalizer saved")

# ─────────────────────────────  ENTRY-POINT  ──────────────────
if __name__ == "__main__":
    mp.freeze_support()                    # Windows-safe
    mp.set_start_method("spawn", force=True)
    main()
