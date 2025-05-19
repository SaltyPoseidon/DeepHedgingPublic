"""
simulator.py – агент-базовый симулятор (расширенная версия).
Изменения:
  • новые поля в SimulatorInfo (см. комментарии);
  • вычисление realized_vol, spread_width, depth_top1;
  • дополнительные греки, implied_vol, book-value и утилита.
Основной цикл simulate / run / liquidate_positions оставлен без изменений.
"""

from __future__ import annotations
from typing import List, Dict, Type, Union
import math, random

import numpy as np
from tqdm import tqdm
from scipy.stats import norm

from AgentBasedModel.exchange import ExchangeAgent, Asset, Option
from AgentBasedModel.traders  import Trader
from AgentBasedModel.extra    import Event
from AgentBasedModel.numerics import safe_exp
from AgentBasedModel.utils.cost_basis import CostBasisFIFO
from AgentBasedModel.numerics import (
    clip_price, safe_div, finite, EPS
)

# ──────────────────────────────────────────────────────────────────────
#  ЧИСТАЯ BS-ЦЕНА и функция для implied σ
# ──────────────────────────────────────────────────────────────────────
def _bs_price(S, K, T, r, sigma, cp: str = "call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if cp == "call" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if cp == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, cp="call", tol=1e-6, n_max=50):
    if price <= 0 or T <= 0:
        return float("nan")
    sigma = 0.2
    for _ in range(n_max):
        px_est = _bs_price(S, K, T, r, sigma, cp)
        vega = (
            S * norm.pdf(
                (math.log(S / K) + (r + 0.5 * sigma**2) * T)
                / (sigma * math.sqrt(T))
            )
            * math.sqrt(T)
        )
        if vega < 1e-8:
            break
        sigma -= (px_est - price) / vega
        if abs(px_est - price) < tol:
            return max(sigma, 1e-8)
    return float("nan")

# ──────────────────────────────────────────────────────────────────────
#  SIMULATOR
# ──────────────────────────────────────────────────────────────────────
class Simulator:
    def __init__(self,
                 assets: List[Asset],
                 exchanges: List[ExchangeAgent],
                 traders: List[Trader],
                 events : List[Event] | None = None):

        self.assets    = assets
        self.exchanges = exchanges
        self.traders   = traders
        self.events    = [ev.link(self) for ev in events] if events else None
        self.info      = SimulatorInfo(exchanges, traders)

    # ------------------------------------------------------------------
    def simulate(self,
                 n_iter: int,
                 silent: bool = False,
                 priority_ids: list[int] | None = None):

        priority_ids = set(priority_ids or [])

        for it in tqdm(range(n_iter), desc="Simulation", disable=silent):
            if self.events:
                for ev in self.events:
                    ev.call(it)

            random.shuffle(self.traders)
            if priority_ids:
                self.traders.sort(key=lambda tr: 0 if tr.id in priority_ids else 1)

            for tr in self.traders:
                if hasattr(tr, "change_strategy"):
                    tr.change_strategy(self.info)
                if hasattr(tr, "change_sentiment"):
                    tr.change_sentiment(self.info)
                tr.call()

            for tr in self.traders:
                tr.income()

            for asset in self.assets:
                asset.update()

            for tr in self.traders:
                if hasattr(tr, "exercise_options"):
                    tr.exercise_options()

            self.info.capture()
        return self

    # ------------------------------------------------------------------
    def liquidate_positions(self):
        """Закрывает остатки позиций по mid-price и фиксирует PnL."""
        for tr in self.traders:
            if isinstance(tr.assets, dict):
                pos_map = tr.assets
            else:
                if hasattr(tr, 'market'):
                    pos_map = {tr.market.id: tr.assets}
                elif hasattr(tr, 'markets'):
                    pos_map = {next(iter(tr.markets)): tr.assets}
                else:
                    continue

            for mid, qty in list(pos_map.items()):
                if qty == 0:
                    continue
                mkt   = self.info.exchanges[mid]
                price = mkt.price()

                cb = tr._cb_trackers.setdefault(mid, CostBasisFIFO())
                if qty > 0:
                    cb.sell(qty, price)
                    tr.cash += price * qty * (1 - mkt.transaction_cost)
                else:
                    cb.buy(-qty, price)
                    tr.cash -= price * (-qty) * (1 + mkt.transaction_cost)

                if isinstance(tr.assets, dict):
                    tr.assets[mid] = 0
                else:
                    tr.assets = 0

                if hasattr(tr, 'orders'):
                    if isinstance(tr.orders, dict):
                        for o in list(tr.orders.get(mid, [])):
                            mkt.cancel_order(o)
                        tr.orders[mid].clear()
                    else:
                        for o in list(tr.orders):
                            mkt.cancel_order(o)
                        tr.orders.clear()

            tr.realized_pnl = sum(cb.realized_pnl for cb in tr._cb_trackers.values())

        self.info.capture()

    # ------------------------------------------------------------------
    def add_market(self,
                   asset: Asset,
                   exchange: ExchangeAgent,
                   traders: List[Trader]):
        """Динамически подключает новый рынок к симулятору."""
        self.assets.append(asset)
        self.exchanges.append(exchange)
        ex_id = exchange.id
        self.info.exchanges[ex_id] = exchange

        # базовые словари
        self.info.prices[ex_id]    = []
        self.info.spreads[ex_id]   = []
        self.info.dividends[ex_id] = []
        self.info.orders[ex_id]    = []

        # новые словари ликвидности / волатильности
        self.info.realized_vol[ex_id] = []
        self.info.spread_width[ex_id] = []
        self.info.depth_top1[ex_id]   = []

        if isinstance(asset, Option):
            self.info.option_metrics[asset.id] = {k: [] for k in
                ("price", "Delta", "Gamma", "Theta", "Vega", "Rho")}
            self.info.tau[asset.id]         = []
            self.info.implied_vol[asset.id] = []

        for tr in traders:
            self.traders.append(tr)
            self.info._register_trader(tr)
            self.info.traders[tr.id] = tr

    # ------------------------------------------------------------------
    def run(self,
            warmup_steps: int,
            main_steps  : int,
            setup_after_warmup=None,
            silent: bool = False):
        if warmup_steps:
            self.simulate(warmup_steps, silent=silent)

        if setup_after_warmup is not None:
            setup_after_warmup(self)
            self.info.capture()

        self.simulate(main_steps, silent=silent)
        self.liquidate_positions()
        return self

# ──────────────────────────────────────────────────────────────────────
#  SIMULATOR INFO  (расширенная версия)
# ──────────────────────────────────────────────────────────────────────
class SimulatorInfo:
    """
    Логирует дополнительно:
      • realized_vol, spread_width, depth_top1
      • τ (time-to-maturity), implied_vol, Vega/Rho
      • агрегированные Delta/Gamma/Vega трейдеров, unrealized_pnl, utility_exp
    """
    # ------------------------------------------------------------
    def __init__(self,
                 exchanges: List[ExchangeAgent],
                 traders  : List[Trader]):

        self.exchanges: Dict[int, ExchangeAgent] = {ex.id: ex for ex in exchanges}
        self.traders  : Dict[int, Trader]        = {tr.id: tr for tr in traders}

        # рынки ─ базовые
        self.prices, self.spreads, self.dividends, self.orders = \
            ({i: [] for i in self.exchanges} for _ in range(4))

        # рынки ─ новые
        self.realized_vol = {i: [] for i in self.exchanges}
        self.spread_width = {i: [] for i in self.exchanges}
        self.depth_top1   = {i: [] for i in self.exchanges}

        # опционы
        self.option_metrics: Dict[int, Dict[str, List[float]]] = {}
        self.tau          : Dict[int, List[float]] = {}
        self.implied_vol  : Dict[int, List[float]] = {}

        # трейдеры ─ базовые
        self.cash, self.assets, self.equities      = {}, {}, {}
        self.returns, self.pnl, self.realized_pnl  = {}, {}, {}

        # трейдеры ─ новые
        self.delta_exp, self.gamma_exp, self.vega_exp = {}, {}, {}
        self.unrealized_pnl, self.utility_exp         = {}, {}

        self.types, self.sentiments = {}, {}

        for tr in traders:
            self._register_trader(tr)

        # параметры
        self.vol_window    = 50
        self.days_per_year = 250
        self.current_gamma : float | None = None

    # ------------------------------------------------------------
    def _register_trader(self, tr: Trader):
        tid = tr.id
        self.cash[tid] = []; self.assets[tid] = []; self.equities[tid] = []
        self.returns[tid] = []; self.pnl[tid] = []; self.realized_pnl[tid] = []

        self.delta_exp[tid] = []; self.gamma_exp[tid] = []; self.vega_exp[tid] = []
        self.unrealized_pnl[tid] = []; self.utility_exp[tid] = []

        self.types[tid] = []; self.sentiments[tid] = []

    # ------------------------------------------------------------
    def capture(self):
        # ── рынки ────────────────────────────────────────────────
        for idx, ex in self.exchanges.items():
            mid = ex.price()
            self.prices[idx].append(mid)
            spr  = ex.spread()
            self.spreads[idx].append(spr)
            self.dividends[idx].append(ex.dividend())

            w = spr["ask"] - spr["bid"] if spr else 0.0
            self.spread_width[idx].append(w)

            depth = 0
            if ex.order_book["bid"]:
                depth += ex.order_book["bid"].first.qty
            if ex.order_book["ask"]:
                depth += ex.order_book["ask"].first.qty
            self.depth_top1[idx].append(depth)

            self.realized_vol[idx].append(
                self.realized_volatility(idx, self.vol_window, self.days_per_year)
            )

        # ── опционы ──────────────────────────────────────────────
        for idx, ex in self.exchanges.items():
            opt = ex.asset
            if not isinstance(opt, Option) or opt.is_expired():
                continue

            metrics = self.bs_option(opt, opt.underlying_exchange.id,
                                     self.vol_window, self.days_per_year)
            if opt.id not in self.option_metrics:
                self.option_metrics[opt.id] = {k: [] for k in metrics}
                self.tau[opt.id]         = []
                self.implied_vol[opt.id] = []

            for k, v in metrics.items():
                self.option_metrics[opt.id][k].append(v)

            self.tau[opt.id].append(opt.expiry / self.days_per_year)

            S = clip_price(opt.underlying_exchange.price())
            K = clip_price(opt.strike)
            T = opt.expiry / self.days_per_year
            r = math.log(1 + opt.underlying_exchange.risk_free_rate) * self.days_per_year
            iv = implied_vol(ex.price(), S, K, T, r, opt.option_type)
            self.implied_vol[opt.id].append(iv if math.isfinite(iv) else float("nan"))

        # ── трейдеры ─────────────────────────────────────────────
        for tid, tr in self.traders.items():
            self.cash[tid].append(tr.cash)
            self.assets[tid].append(
                tr.assets.copy() if isinstance(tr.assets, dict) else tr.assets
            )
            eq = tr.equity()
            self.equities[tid].append(eq)

            if len(self.equities[tid]) > 1 and self.equities[tid][-2] != 0:
                ret = safe_div(eq - self.equities[tid][-2], self.equities[tid][-2])
                pnl = eq - self.equities[tid][-2]
            else:
                ret = pnl = 0.0
            self.returns[tid].append(ret); self.pnl[tid].append(pnl)
            self.realized_pnl[tid].append(tr.realized_pnl)

            # агрегированные риски + unrealized PnL
            d_tot = g_tot = v_tot = 0.0
            unrl  = 0.0
            if isinstance(tr.assets, dict):
                pos_iter = tr.assets.items()
            else:
                pos_iter = [(next(iter(tr.markets)), tr.assets)]

            for mid, qty in pos_iter:
                if qty == 0:
                    continue
                mkt = self.exchanges[mid]
                px  = mkt.price()
                if isinstance(mkt.asset, Option):
                    opt  = mkt.asset
                    mets = self.option_metrics.get(opt.id, {})
                    d_tot += mets.get("Delta", [0])[-1] * qty
                    g_tot += mets.get("Gamma", [0])[-1] * qty
                    v_tot += mets.get("Vega",  [0])[-1] * qty
                else:
                    d_tot += qty
                unrl += qty * px

            self.delta_exp[tid].append(d_tot)
            self.gamma_exp[tid].append(g_tot)
            self.vega_exp[tid].append(v_tot)
            self.unrealized_pnl[tid].append(unrl)

            γ = self.current_gamma or 0.0
            arg = -γ * eq
            util = -safe_exp(arg) if γ > 0 else -eq

            self.types[tid].append(tr.type)
            if hasattr(tr, "sentiment"):
                self.sentiments[tid].append(tr.sentiment)

    # ------------------------------------------------------------
    #  Реализованная волатильность
    # ------------------------------------------------------------
    def realized_volatility(self,
                            idx            : int,
                            window         : int  = 50,
                            days_per_year  : int  = 250):
        prices = np.asarray(self.prices[idx][-window - 1:], dtype=float)
        if prices.size < window + 1:
            return 0.0
        prices = np.clip(prices, 0.001 * np.median(prices), None)
        log_ret = np.diff(np.log(prices))
        sigma = np.nan_to_num(log_ret.std(ddof=1), nan=0.0)
        return float(sigma) * math.sqrt(days_per_year)

    # ------------------------------------------------------------
    #  Black–Scholes (расширенный, с Vega / Rho)
    # ------------------------------------------------------------
    def bs_option(self,
                  option       : Option,
                  idx_under    : int,
                  window       : int = 50,
                  days_in_year : int = 250) -> dict:

        ex = self.exchanges[idx_under]
        S  = clip_price(ex.price())
        K  = clip_price(option.strike)
        T  = option.expiry / days_in_year
        r  = math.log(1 + ex.risk_free_rate) * days_in_year

        if T <= 0:
            payoff = max(S - K, 0) if option.option_type == "call" else max(K - S, 0)
            base = {"price": payoff}
            return {**base, "Delta": 0.0, "Gamma": 0.0,
                    "Theta": 0.0, "Vega": 0.0, "Rho": 0.0}

        sigma = self.realized_volatility(idx_under, window, days_in_year)
        if sigma == 0:
            return {k: 0.0 for k in ("price", "Delta", "Gamma", "Theta", "Vega", "Rho")}

        d1 = safe_div(math.log(S / K) + (r + 0.5 * sigma ** 2) * T,
                      sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option.option_type == "call":
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            rho   = K * T * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            rho   = -K * T * math.exp(-r * T) * norm.cdf(-d2)

        gamma = norm.pdf(d1) / max(S * sigma * math.sqrt(T), EPS)
        common = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
        theta = (common - r * K * math.exp(-r * T) * norm.cdf(d2)
                 if option.option_type == "call"
                 else common + r * K * math.exp(-r * T) * norm.cdf(-d2))
        theta /= days_in_year
        vega = S * norm.pdf(d1) * math.sqrt(T)

        return {
            "price": finite(price, 0.0),
            "Delta": finite(delta, 0.0),
            "Gamma": finite(gamma, 0.0),
            "Theta": finite(theta, 0.0),
            "Vega":  finite(vega, 0.0),
            "Rho" :  finite(rho, 0.0),
        }
