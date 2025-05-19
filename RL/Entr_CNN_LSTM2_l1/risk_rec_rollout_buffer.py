# risk_rec_rollout_buffer.py
import numpy as np
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer



class RiskRecRolloutBuffer(RecurrentRolloutBuffer):
    def __init__(self, *args, lambda_fixed=1e-3, **kw):
        super().__init__(*args, **kw)
        self.lambda_fixed = lambda_fixed

    def compute_returns_and_advantage(self, last_values, dones):
        last_v  = last_values.cpu().numpy().flatten()
        z_next  = np.exp(-self.lambda_fixed * last_v)  # z_{T}

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                non_term = 1.0 - dones.astype(np.float32)
            else:
                non_term = 1.0 - self.episode_starts[step + 1]

            r_t = self.rewards[step]
            z_t = np.exp(-self.lambda_fixed * r_t) * \
                  (z_next ** (self.gamma * non_term))
            v_targ = -np.log(np.clip(z_t, 1e-8, 1.0)) / self.lambda_fixed

            self.returns[step]    = v_targ
            self.advantages[step] = v_targ - self.values[step]
            z_next = z_t
