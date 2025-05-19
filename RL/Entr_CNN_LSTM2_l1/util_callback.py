# util_callback.py
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

LAMBDA = 1          # тот же λ, что и в критике/среде

class UtilityCallback(BaseCallback):
    """
    Каждые `eval_freq` шагов запускает N тестовых эпизодов,
    считает полезность U и пишет в лог.
    """
    def __init__(self, eval_env, n_eval_episodes=10, eval_freq=10_000):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.counter = 0

    def _on_step(self) -> bool:
        self.counter += 1
        if self.counter % self.eval_freq != 0:
            return True

        utils = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=None)
            done, trunc = False, False
            while not (done or trunc):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = self.eval_env.step(action)

            # info может хранить PnL; если нет — берем из среды
            pnl = info.get("pnl_final", self.eval_env.agent.realized_pnl)
            utils.append(-np.exp(-LAMBDA * pnl))

        mean_u = float(np.mean(utils))
        self.logger.record("eval/utility", mean_u)
        print(f"[Eval] mean U = {mean_u:.4f} ---------------------------------------------")
        return True
