# action_logger.py
from __future__ import annotations
import numpy as np
from collections import Counter, defaultdict
from stable_baselines3.common.callbacks import BaseCallback

# --- расшифровка MultiDiscrete([4, qty, side, bin, idx]) -------------
ACT_MAP = {
    0: "pass",
    1: "market",
    2: "limit",
    3: "cancel",
}
SIDE = {0: "buy", 1: "sell"}


class ActionLoggerCallback(BaseCallback):
    """
    Каждые `log_freq` шагов:
      • собирает последние действия всех env;
      • пишет распределение действий, средний raw-PnL, позицию.
    """

    def __init__(self, log_freq: int = 2_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.buff: list[np.ndarray] = []        # (n_envs, 5) actions
        self.pnl_raw: list[float] = []
        self.pos_raw: list[float] = []

    # --- SB3 хук ------------------------------------------------------
    def _on_step(self) -> bool:
        # info из vecenv – берем final_PnL и позицию, если есть
        for info in self.locals.get("infos", []):
            if "raw_reward" in info:
                self.pnl_raw.append(info["raw_reward"])
            if "pos_S" in info:
                self.pos_raw.append(info["pos_S"])

        self.buff.append(self.locals["actions"].copy())
        if len(self.buff) < self.log_freq:
            return True

        # ---- агрегируем ---------------------------------------------
        flat = np.concatenate(self.buff)
        cnt: Counter[str] = Counter()
        for row in flat:
            kind = ACT_MAP[row[0]]
            if kind in ("market", "limit"):
                kind = f"{kind}-{SIDE[row[2]]}"
            cnt[kind] += 1

        total = sum(cnt.values()) or 1
        fracs = {k: v / total for k, v in cnt.items()}

        # --- лог в TensorBoard / stdout ------------------------------
        for k, v in fracs.items():
            self.logger.record(f"acts/{k}", v)
        if self.pnl_raw:
            self.logger.record("diag/mean_raw_pnl", np.mean(self.pnl_raw))
        if self.pos_raw:
            self.logger.record("diag/max_pos_S", np.max(self.pos_raw))

        if self.verbose:
            print(f"[ActLog] {fracs}  "
                  f"meanPnL={np.mean(self.pnl_raw):.3f}  "
                  f"maxPos={np.max(self.pos_raw):.1f}")

        # очистка
        self.buff.clear()
        self.pnl_raw.clear()
        self.pos_raw.clear()
        return True
