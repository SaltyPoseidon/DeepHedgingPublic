# save_best_utility.py
import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveBestUtilityCallback(BaseCallback):
    """
    Сохраняет модель и VecNormalize при росте eval/utility.
    """
    def __init__(self, check_freq: int, save_dir: str, vec_norm_env, verbose=0):
        super().__init__(verbose)
        self.check_freq   = check_freq
        self.save_dir     = save_dir
        self.vec_norm_env = vec_norm_env
        self.best_util    = -float("inf")
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # раз в check_freq шагов смотрим, не появилась ли новая лучшая utility
        if self.n_calls % self.check_freq != 0:
            return True

        util = self.logger.name_to_value.get("eval/utility")
        if util is None:
            return True                        # ещё нет метрики

        if util > self.best_util:
            self.best_util = util
            path_model = os.path.join(self.save_dir, "best_model")
            path_norm  = os.path.join(self.save_dir, "vecnorm_best.pkl")
            self.model.save(path_model)
            if self.vec_norm_env is not None:
                self.vec_norm_env.save(path_norm)
            if self.verbose:
                print(f"▼ new best utility {util:.4f} ⇒ saved to {path_model}")
        return True
