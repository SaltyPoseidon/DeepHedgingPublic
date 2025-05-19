# risk_rec_ppo.py
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from risk_rec_rollout_buffer import RiskRecRolloutBuffer

class RiskRecPPO(RecurrentPPO):
    def __init__(self, *args, lambda_fixed=1e-3, **kw):
        self.lambda_fixed = lambda_fixed
        super().__init__(*args, **kw)

    def _init_rollout_buffer(self):
        self.rollout_buffer = RiskRecRolloutBuffer(
            self.n_steps, self.observation_space, self.action_space,
            self.device, gamma=self.gamma, gae_lambda=self.gae_lambda,
            n_envs=self.n_envs, lambda_fixed=self.lambda_fixed
        )
