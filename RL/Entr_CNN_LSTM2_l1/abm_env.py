"""
abm_env.py – лёгкая версия с MARKET / LIMIT / CANCEL
─────────────────────────────────────────────────────
* action_space = MultiDiscrete([4, Q_MAX+1, 2, N_PRICEBIN, MAX_OPEN])
    0  • PASS
    1  • MARKET      (qty , side)
    2  • LIMIT       (qty , side , price_bin)
    3  • CANCEL      (index)
* observation = 16 признаков
    (13 прежних + n_bid, n_ask, mean_age)
"""

from __future__ import annotations
import secrets
import math, random, time
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from AgentBasedModel import Stock, ExchangeAgent, Simulator, OptionExchangeAgent
from AgentBasedModel.exchange import Option
from AgentBasedModel.multitrader import (
    MultiAssetRandomTrader, MultiAssetFundamentalist,
    MultiAssetChartist2D, MultiAssetMarketMaker2D,
)

from AgentBasedModel.lob_features import build_lob_features

from AgentBasedModel.optiontraders import (
    OptionTrader, RandomOptionTrader, FairValueOptionTrader,
    SentimentOptionTrader, DeltaHedger, FairPriceMarketMaker,
)
from AgentBasedModel.utils.cost_basis import CostBasisFIFO
from AgentBasedModel.utils.orders import Order   # ← нужен для open_orders

# ─── базовые параметры сцены ──────────────────────────────────────────
RFR, BASE_PRICE = 5e-5, 100.0
DIVIDEND        = BASE_PRICE * RFR
N_WARMUP, N_MAIN = 100, 300

TR_CNT_BASE = dict(random=20, fundamentalist=20,
                   chartist=20, market_maker=4)
TR_CNT_OPT  = dict(random_option=10, fair_value=20,
                   sentiment=20, delta_hedger=2, fair_mm=40)

INIT_CASH  = 1_000.0
Q_MAX      = 5             # макс. объём за действие
N_PRICEBIN = 9             # −4 … +4 тика
MAX_OPEN   = 8             # максимум висящих собственных лимитов

KAPPA      = 0.25
SIDE_OPT   = -1            # long(+1) / short(−1)

LAMBDA_FIXED = 1

# ───────────── reset ─────────────
MIN_CALL_START = 0.25      # $ – минимальная начальная цена опциона
MAX_WARMUP_RETRY = 10      # перестраховка: чтобы не застрять в while
# ───────────────────────────────────────────────────────────────

BASE_SEED_WARM = 156567     # фиксированный сид для warm-up


# 400 - 4.21
# 156567 - 6.132

K_LEVELS   = 100        # ← сколько ценовых уровней берём в CNN
LOB_LEN    = 2 * K_LEVELS * 2      # 400 при K=100
PORT_FEAT  = 16
OBS_DIM    = LOB_LEN + PORT_FEAT    # 416

def reseed_global(seed: int):
    """
    Пересевает стандартные генераторы `random` и `numpy`.
    Подходит как для фиксированного warm-up, так и для «живой» части.
    """
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

# ╭────────────────────────────────────────────────────────────╮
# │  RL-трейдер (market + limit + cancel)                      │
# ╰────────────────────────────────────────────────────────────╯
class RLTrader(OptionTrader):
    def __init__(self, markets, sim_info, *, cash=INIT_CASH):
        super().__init__(markets, cash)
        self.sim_info     = sim_info
        self.pending: dict | None = None
        self.open_orders: List[Order] = []
        self.type = "RLTrader"

        # заполним позже в reset()
        self.mid_S: int | None = None

    # ───── интерфейс «задать действие» ───────────────────────
    def set_action(self, cmd: dict | None):         # None → PASS
        self.pending = cmd

    # ───── главный вызов трейдера ────────────────────────────
    def call(self):
        self.exercise_options()
        if self.pending is None:
            return
        cmd, self.pending = self.pending, None
        if self.mid_S is None:
            # должно быть установлено reset()-ом
            self.mid_S = next(mid for mid, m in self.markets.items()
                              if m.asset.type == "Stock")
        mid = self.mid_S

        act = cmd["act"]
        if act == "mkt":
            fn = self._buy_market if cmd["side"] > 0 else self._sell_market
            fn(mid, cmd["qty"])

        elif act == "lmt":
            fn = self._buy_limit if cmd["side"] > 0 else self._sell_limit
            order = fn(mid, cmd["qty"], cmd["price"])
            if order:
                self.open_orders.append(order)
                if len(self.open_orders) > MAX_OPEN:
                    self.open_orders.pop(0)     # старейший

        elif act == "cxl" and self.open_orders:
            idx = min(cmd["index"], len(self.open_orders) - 1)
            self._cancel_order(mid, self.open_orders[idx])
            self.open_orders.pop(idx)

# ╭────────────────────────────────────────────────────────────╮
# │  Gym-окружение                                             │
# ╰────────────────────────────────────────────────────────────╯
class ABMDeepHedgeEnv(gym.Env):
    """Deep-Hedging окружение: MARKET, LIMIT, CANCEL."""

    metadata: Dict[str, str] = {}

    # ───── new ACTION SPACE ──────────────────────────────────
    action_space = spaces.MultiDiscrete([
        4,            # 0=PASS 1=MARKET 2=LIMIT 3=CANCEL
        Q_MAX + 1,    # qty
        2,            # side 0=BUY 1=SELL
        N_PRICEBIN,   # price_bin 0..8
        MAX_OPEN      # index для cancel
    ])

    # 16-мерное наблюдение
    observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

    # ───── ctor ──────────────────────────────────────────────
    def __init__(self, *, lambda_fixed: float = LAMBDA_FIXED, seed: int | None = None):
        super().__init__()
        # фиксированный warm-up сид (constant)
        reseed_global(BASE_SEED_WARM)

        self._base_seed_episode = seed

        self.lambda_fixed = lambda_fixed
        self.sim:   Simulator | None = None
        self.agent: RLTrader  | None = None
        self.prev_rpnl = 0.0
        self.ticks_left = 0

        # вспомогательные id
        self.mid_S: int | None = None
        self.mid_C: int | None = None

    def _seed(self, seed: int):
        """
        Сохраняет seed для «живой» части и пересеивает глобальные RNG
        (используется в reset, можно дёргать вручную).
        """
        self._base_seed_episode = seed
        reseed_global(seed)

    # ───── reset ─────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._seed(seed)

        self.sim, self.agent = self._build_simulator()
        self.sim.info.current_gamma = self.lambda_fixed

        # cache id-рынков
        self.mid_S = next(mid for mid, e in self.sim.info.exchanges.items()
                          if e.asset.type == "Stock")
        self.mid_C = next(mid for mid, e in self.sim.info.exchanges.items()
                          if e.asset.type == "Option")
        self.agent.mid_S = self.mid_S

        self.prev_rpnl   = self.sim.info.realized_pnl[self.agent.id][-1]
        self.ticks_left  = N_MAIN
        return self._obs(), {}

    # ── step ─────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        self.agent.set_action(self._decode(action))
        self.sim.simulate(1, silent=True, priority_ids=[self.agent.id])
        self.ticks_left -= 1

        rpnl_now = self.sim.info.realized_pnl[self.agent.id][-1]
        reward = rpnl_now - self.prev_rpnl
        self.prev_rpnl = rpnl_now

        terminated = (self.ticks_left == 0) or self._option_expired()
        truncated = False

        info = {}
        final_pnl = 0
        if terminated:
            # ликвидация и финальный PnL
            self.sim.liquidate_positions()
            rpnl_liq = self.sim.info.realized_pnl[self.agent.id][-1]
            reward += rpnl_liq - self.prev_rpnl
            self.prev_rpnl = rpnl_liq
            final_pnl = float(self.sim.info.realized_pnl[self.agent.id][-1])
            # --- единый канал для VecEnv/Monitor ---
            info = {
                "final_PnL": final_pnl,  # корневой ключ
                "episode": {  # ← то, что читает Monitor
                    "r": final_pnl,
                    "l": N_MAIN,
                    "final_PnL": final_pnl
                }
            }
            reward_scaled = reward / 100
            return self._obs(), float(reward_scaled), terminated, truncated, info

        reward_scaled = reward /100

        return self._obs(), float(reward_scaled), terminated, truncated, info

    # ───── render (консоль) ──────────────────────────────────
    def render(self, mode="human"):
        if not self.sim:
            return
        print(
            f"T={N_MAIN - self.ticks_left:3d} | "
            f"S={self.sim.info.prices[self.mid_S][-1]:.2f} | "
            f"Cash={self.agent.cash:.1f}"
        )

    def close(self): ...

    # ───── helpers ───────────────────────────────────────────
    @staticmethod
    def _seed(seed: int):
        random.seed(seed); np.random.seed(seed)

    def _decode(self, vec: np.ndarray):
        """MultiDiscrete → dict команды или None."""
        atype, qty, side, pbin, idx = map(int, vec)

        # 0 = PASS  или qty=0
        if atype == 0 or qty == 0:
            return None

        if atype == 1:      # MARKET
            return dict(act="mkt",
                        qty=qty,
                        side=+1 if side == 0 else -1)

        if atype == 2:      # LIMIT
            mid  = self.sim.info.prices[self.mid_S][-1]
            spr  = self.sim.info.spreads[self.mid_S][-1]
            tick = spr["ask"] - spr["bid"]
            price = mid + (pbin - (N_PRICEBIN // 2)) * tick
            return dict(act="lmt",
                        qty=qty,
                        side=+1 if side == 0 else -1,
                        price=max(price, 1e-6))

        if atype == 3:      # CANCEL
            return dict(act="cxl", index=idx)

        return None   # fallback

    def _option_expired(self) -> bool:
        return self.sim.info.exchanges[self.mid_C].asset.is_expired()

    # ═══════════════════════════════════════════════════════════
    #  Создание симулятора
    # ═══════════════════════════════════════════════════════════
    def _build_simulator(self):
        """Создаём базовый рынок и опцион ТОЛЬКО если call не ≈0."""

        reseed_global(BASE_SEED_WARM)

        for _try in range(MAX_WARMUP_RETRY):
            # 1) базовый рынок и warm-up
            stock = Stock(DIVIDEND)
            stock_mkt = ExchangeAgent(stock, RFR, transaction_cost=0.01)
            base_traders = (
                    [MultiAssetRandomTrader([stock_mkt]) for _ in range(TR_CNT_BASE["random"])] +
                    [MultiAssetFundamentalist([stock_mkt], access=1) for _ in range(TR_CNT_BASE["fundamentalist"])] +
                    [MultiAssetChartist2D([stock_mkt]) for _ in range(TR_CNT_BASE["chartist"])] +
                    [MultiAssetMarketMaker2D([stock_mkt]) for _ in range(TR_CNT_BASE["market_maker"])]
            )
            sim = Simulator([stock], [stock_mkt], base_traders)
            sim.simulate(N_WARMUP, silent=True)

            # 2) смотрим, сколько стоит call при текущем S
            S_last = stock_mkt.price()
            bs_price = sim.info.bs_option(
                option=Option(stock, 100, N_MAIN, "call", stock_mkt),
                idx_under=stock_mkt.id,
                window=50, days_in_year=250)["price"]

            if bs_price >= MIN_CALL_START:
                break  # условие выполнено
        else:
            # даже после MAX_WARMUP_RETRY call дешёвый – берём как есть
            print("⚠  Call price < threshold – продолжаем с дешёвым опционом")



        # 3) строим опционный рынок (как было)
        option = Option(stock, 100.0, N_MAIN, "call", stock_mkt)
        option_mkt = OptionExchangeAgent(stock_mkt, option,
                                         sim_info=sim.info,
                                         transaction_cost=0.01)

        opt_traders = (
            [RandomOptionTrader([stock_mkt, option_mkt])              for _ in range(TR_CNT_OPT["random_option"])] +
            [FairValueOptionTrader([stock_mkt, option_mkt], sim.info) for _ in range(TR_CNT_OPT["fair_value"])] +
            [SentimentOptionTrader([stock_mkt, option_mkt], sim.info) for _ in range(TR_CNT_OPT["sentiment"])] +
            [DeltaHedger([stock_mkt, option_mkt], sim.info)           for _ in range(TR_CNT_OPT["delta_hedger"])] +
            [FairPriceMarketMaker([stock_mkt, option_mkt], sim.info)  for _ in range(TR_CNT_OPT["fair_mm"])]
        )

        rl = RLTrader([stock_mkt, option_mkt], sim.info)
        opt_traders.append(rl)
        sim.add_market(option, option_mkt, opt_traders)

        # ─ стартовая позиция по опциону ─
        price_series = sim.info.option_metrics[option.id]["price"]
        fair = price_series[-1] if price_series else sim.info.bs_option(option, stock_mkt.id)["price"]
        fair = max(fair, 0.1)
        qty0 = max(1, math.ceil(KAPPA * INIT_CASH / fair))
        mid_C = option_mkt.id
        rl.assets[mid_C] = rl.assets.get(mid_C, 0) + SIDE_OPT * qty0
        rl.cash         -= SIDE_OPT * qty0 * fair
        cb = rl._cb_trackers.setdefault(mid_C, CostBasisFIFO())
        (cb.buy if SIDE_OPT > 0 else cb.sell)(qty0, fair)

        sim.info.capture()

        # 2) «живой» сид (если None — генерим по времени)
        live_seed = secrets.randbits(32)
        #
        # live_seed = (
        #     self._base_seed_episode
        #     if self._base_seed_episode is not None
        #     else int(time.time() * 1e6) & 0xFFFFFFFF
        # )
        #
        reseed_global(live_seed)

        return sim, rl

    # ═══════════════════════════════════════════════════════════
    #  Observation (16 features)
    # ═══════════════════════════════════════════════════════════
    def _obs(self):
        inf = self.sim.info
        mid_S = self.mid_S
        mid_C = self.mid_C

        # ───────── LOB-тензор → вектор (400 фич при K=100) ──────────
        exch = inf.exchanges[mid_S]
        lob_t = build_lob_features(exch.order_book, K=K_LEVELS, tick=0.01)
        lob_vec = lob_t.flatten()  # shape = (LOB_LEN,)

        # ───────── портфельные и опционные признаки ─────────────────
        opt = inf.exchanges[mid_C].asset
        S = inf.prices[mid_S][-1]
        P_call = inf.prices[mid_C][-1]
        spread_S = inf.spreads[mid_S][-1]["ask"] - inf.spreads[mid_S][-1]["bid"]
        spread_C = inf.spreads[mid_C][-1]["ask"] - inf.spreads[mid_C][-1]["bid"]
        sigma = inf.realized_volatility(mid_S, 50)
        tau = opt.expiry / N_MAIN
        strike = opt.strike

        mets = inf.option_metrics.get(opt.id, {})
        d_opt = mets.get("Delta", [0.0])[-1]
        g_opt = mets.get("Gamma", [0.0])[-1]
        t_opt = mets.get("Theta", [0.0])[-1]

        pos_S = self.agent.assets.get(mid_S, 0)
        pos_C = self.agent.assets.get(mid_C, 0)
        cash_n = self.agent.cash / INIT_CASH

        delta_port = pos_S + pos_C * d_opt
        gamma_port = pos_C * g_opt
        theta_port = pos_C * t_opt

        # ───────── сведения о собственных лимит-ордерах ─────────────
        n_bid = sum(1 for o in self.agent.open_orders if o.order_type == "bid")
        n_ask = len(self.agent.open_orders) - n_bid
        mean_age = 0.0
        if self.agent.open_orders:
            mean_age = np.mean([self.ticks_left / N_MAIN for _ in self.agent.open_orders])

        port_vec = np.array([
            S, P_call,
            spread_S, spread_C,
            sigma, tau, strike,
            delta_port, gamma_port, theta_port,
            pos_S, pos_C, cash_n,
            n_bid / MAX_OPEN, n_ask / MAX_OPEN, mean_age
        ], dtype=np.float32)  # ← 16 чисел

        # ───────── объединяем LOB + портфель ────────────────────────
        obs = np.concatenate([lob_vec, port_vec])  # длина = 416

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
