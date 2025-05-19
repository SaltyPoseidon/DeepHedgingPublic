from __future__ import annotations
from typing import List, Dict
import random, math

from AgentBasedModel.multitrader import MultiAssetTrader
from AgentBasedModel.exchange    import ExchangeAgent, Option
from AgentBasedModel.utils.orders import Order
import numpy as np

from AgentBasedModel.utils.cost_basis import CostBasisFIFO

# ──────────────────────────────────────────────────────────────────────
#  Helpers & numeric‑safety utilities
# ──────────────────────────────────────────────────────────────────────

_MIN_PRICE = 1e-6           # floor to avoid zero / negative prices
_EPS       = 1e-9           # tiny epsilon for comparisons


def _isfinite(x: float) -> bool:
    """True for real numbers that are not NaN / ±inf."""
    return math.isfinite(x) and not math.isnan(x)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rand_qty(a: int = 1, b: int = 5) -> int:
    return random.randint(a, b)


def _inside(spread: dict) -> float:
    """Draw a price inside bid‑ask; fall back to _MIN_PRICE if book empty."""
    if not spread:
        return _MIN_PRICE
    return max(random.uniform(spread['bid'], spread['ask']), _MIN_PRICE)


def _outside(side: str, spread: dict, scale: float = 2.5) -> float:
    if not spread:
        return _MIN_PRICE
    delta = random.expovariate(1 / scale)
    p = spread['bid'] - delta if side == 'bid' else spread['ask'] + delta
    return max(p, _MIN_PRICE)


# ──────────────────────────────────────────────────────────────────────
#  Base: common logic + auto‑exercise
# ──────────────────────────────────────────────────────────────────────

class OptionTrader(MultiAssetTrader):
    """Common superclass for option strategies (handles expiry)."""

    # ------------- expiry --------------------------------------------------
    def exercise_options(self):
        """
        • проверяет все опционы;
        • рассчитывает pay-off;
        • закрывает позицию через FIFO-трекинг (sell/cover по цене pay-off);
        • апдейтит cash и self.realized_pnl;
        • очищает остаточные ордера.
        """
        for mid, mkt in list(self.markets.items()):
            opt = mkt.asset
            if not isinstance(opt, Option) or not opt.is_expired():
                continue                       # живой опцион — пропускаем

            pos = self.assets.get(mid, 0)
            if pos == 0:
                continue                       # нет позиции

            payoff = opt.exercise() or 0.0      # float, ≥ 0
            if not _isfinite(payoff) or payoff < 0:
                payoff = 0.0

            # ► FIFO-закрытие позиции
            cb = self._cb_trackers.setdefault(mid, CostBasisFIFO())
            if pos > 0:           # long → «продажа» по цене payoff
                cb.sell(pos, payoff)
            else:                 # short → «покупка» для покрытия
                cb.buy(-pos, payoff)
            # пересчёт суммарного реализованного PnL
            self.realized_pnl = sum(c.realized_pnl for c in self._cb_trackers.values())

            # ► Денежный поток
            self.cash += payoff * pos          # для short pos<0 → outflow

            # ► Обнуляем позицию и удаляем ордера
            self.assets[mid] = 0
            for o in list(self.orders[mid]):
                self._cancel_order(mid, o)
    # ------------- helper --------------------------------------------------
    @staticmethod
    def qty_from_misprice(rel: float, k: float = 10, qmax: int = 5) -> int:
        """Convert relative mis‑pricing to discrete order size."""
        if not _isfinite(rel):
            return 1
        q = int(abs(rel) * k) + 1
        return _clamp(q, 1, qmax)

    # ----------------------------------------------------------------------
    def call(self) -> None:  # pragma: no cover
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────
#  1. Random noise trader
# ──────────────────────────────────────────────────────────────────────

class RandomOptionTrader(OptionTrader):
    def __init__(self, markets: List[ExchangeAgent], cash: float = 1_000):
        super().__init__(markets, cash)
        self.type = 'RandomOptionTrader'

    def call(self) -> None:
        self.exercise_options()
        live = [mid for mid, m in self.markets.items() if not (isinstance(m.asset, Option) and m.asset.is_expired())]
        if not live:
            return
        mid = random.choice(live)
        mkt = self.markets[mid]
        spread = mkt.spread()

        r = random.random()
        if r < 0.15:                                   # market order
            (self._buy_market if random.random() < .5 else self._sell_market)(mid, _rand_qty())
        elif r < 0.5:                                  # limit inside
            price = _inside(spread)
            (self._buy_limit if random.random() < .5 else self._sell_limit)(mid, _rand_qty(), price)
        elif r < 0.85 and self.orders[mid]:            # cancel
            self._cancel_order(mid, random.choice(self.orders[mid]))


# ──────────────────────────────────────────────────────────────────────
#  2. Fair‑value (Black‑Scholes) trader
# ──────────────────────────────────────────────────────────────────────

class FairValueOptionTrader(OptionTrader):
    def __init__(self, markets: List[ExchangeAgent], sim_info, *, cash: float = 1_000, tol: float = 2e-3):
        super().__init__(markets, cash)
        self.sim_info = sim_info
        self.tol = tol
        self.type = 'FairValueOptionTrader'

    def call(self) -> None:
        self.exercise_options()
        for mid, mkt in self.markets.items():
            opt = mkt.asset
            if not isinstance(opt, Option) or opt.is_expired():
                continue
            metrics = self.sim_info.option_metrics.get(opt.id)
            if not metrics or not metrics['price']:
                continue
            fair = metrics['price'][-1]
            if not _isfinite(fair) or fair <= 0:
                continue
            mprice = mkt.price()
            if not _isfinite(mprice) or mprice <= 0:
                continue
            fair_safe = np.maximum(np.abs(fair), 1e-8)
            mis = np.clip((fair - mprice) / fair_safe, -1.0, 1.0)
            if not _isfinite(mis) or abs(mis) < self.tol:
                continue
            mis = _clamp(mis, -1.0, 1.0)
            qty = self.qty_from_misprice(mis)
            spread = mkt.spread()
            if mis > 0:  # undervalued → buy
                price = min(fair, spread['ask']) if spread else fair
                self._buy_limit(mid, qty, max(price, _MIN_PRICE))
            else:        # overvalued → sell
                price = max(fair, spread['bid']) if spread else fair
                self._sell_limit(mid, qty, max(price, _MIN_PRICE))


# ──────────────────────────────────────────────────────────────────────
#  3. Momentum / sentiment trader
# ──────────────────────────────────────────────────────────────────────

class SentimentOptionTrader(OptionTrader):
    def __init__(self, markets: List[ExchangeAgent], sim_info, *, cash: float = 1_000, lookback: int = 3):
        super().__init__(markets, cash)
        self.sim_info  = sim_info
        self.lookback  = max(1, lookback)
        self.sentiment = 'Bullish' if random.random() > .5 else 'Bearish'
        self.type      = 'SentimentOptionTrader'

    def _update(self, idx_under: int) -> None:
        prices = self.sim_info.prices[idx_under]
        if len(prices) < self.lookback + 1:
            return
        prev = prices[-1 - self.lookback]
        curr = prices[-1]
        if not (_isfinite(prev) and _isfinite(curr)):
            return
        if curr > prev + _EPS:
            self.sentiment = 'Bullish'
        elif curr < prev - _EPS:
            self.sentiment = 'Bearish'

    def call(self) -> None:
        self.exercise_options()
        for mid, mkt in self.markets.items():
            opt = mkt.asset
            if not isinstance(opt, Option) or opt.is_expired():
                continue
            self._update(opt.underlying_exchange.id)
            price = _inside(mkt.spread())
            qty   = _rand_qty()
            if self.sentiment == 'Bullish':
                self._buy_limit(mid, qty, price)
            else:
                self._sell_limit(mid, qty, price)


# ──────────────────────────────────────────────────────────────────────
#  4. Delta‑hedger
# ──────────────────────────────────────────────────────────────────────

class DeltaHedger(OptionTrader):
    def __init__(
        self,
        markets: List[ExchangeAgent],
        sim_info,
        *,
        fraction: float = 0.1,
        target_delta: float = 0.0,
        hedge_tolerance: float = 1e-3,
        prob_market: float = 0.7,
        prob_cancel: float = 0.2,
        max_per_trade: int = 10,
        cash: float = 1_000,
    ):
        super().__init__(markets, cash)
        self.sim_info      = sim_info
        self.fraction      = fraction
        self.target_delta  = target_delta
        self.hedge_tol     = hedge_tolerance
        self.prob_market   = prob_market
        self.prob_cancel   = prob_cancel
        self.max_per_trade = max(1, max_per_trade)
        self._initialized  = False
        self.type          = 'DeltaHedger'

    # ------------------------------------------------------------------
    def _cancel_some_under_orders(self, under_mid: int):
        for o in list(self.orders.get(under_mid, [])):
            if random.random() < self.prob_cancel:
                self._cancel_order(under_mid, o)

    # ------------------------------------------------------------------
    def call(self) -> None:
        opt_mid   = next((mid for mid, m in self.markets.items() if isinstance(m.asset, Option)), None)
        under_mid = next((mid for mid, m in self.markets.items() if not isinstance(m.asset, Option)), None)
        if opt_mid is None or under_mid is None:
            return
        self.exercise_options()
        self._cancel_some_under_orders(under_mid)

        metrics = self.sim_info.option_metrics.get(self.markets[opt_mid].asset.id, {})
        if not metrics or not metrics['price'] or not metrics['Delta']:
            return
        fair_px   = metrics['price'][-1]
        fair_delta= metrics['Delta'][-1]
        if not (_isfinite(fair_px) and fair_px > 0 and _isfinite(fair_delta)):
            return

        # --- initial setup: short option + buy Δ shares -----------------
        if not self._initialized:
            qty_opt = max(1, int(self.cash * self.fraction / fair_px))
            self._sell_limit(opt_mid, qty_opt, fair_px)
            hedge_qty = max(0, math.ceil(qty_opt * fair_delta))
            if hedge_qty:
                if random.random() < self.prob_market:
                    self._buy_market(under_mid, hedge_qty)
                else:
                    price = max(self.markets[under_mid].spread()['bid'] + 0.1, _MIN_PRICE)
                    self._buy_limit(under_mid, hedge_qty, price)
            self._initialized = True
            return

        # --- ongoing re‑hedge ------------------------------------------
        pos_opt   = self.assets.get(opt_mid, 0)
        pos_under = self.assets.get(under_mid, 0)
        net_delta = pos_opt * fair_delta + pos_under
        if not _isfinite(net_delta):
            return

        diff = net_delta - self.target_delta
        if abs(diff) < self.hedge_tol:
            return

        hedge_qty = min(self.max_per_trade, math.ceil(abs(diff)))
        if hedge_qty <= 0:
            return

        if random.random() < self.prob_market:           # market execution
            if diff > 0:
                self._sell_market(under_mid, hedge_qty)
            else:
                self._buy_market(under_mid, hedge_qty)
        else:                                            # aggressive limit
            sp = self.markets[under_mid].spread()
            if not sp:
                return
            if diff > 0:
                price = max(sp['ask'] - 0.1, _MIN_PRICE)
                self._sell_limit(under_mid, hedge_qty, price)
            else:
                price = max(sp['bid'] + 0.1, _MIN_PRICE)
                self._buy_limit(under_mid, hedge_qty, price)


# ──────────────────────────────────────────────────────────────────────
#  5. Fair‑price market maker
# ──────────────────────────────────────────────────────────────────────

class FairPriceMarketMaker(OptionTrader):
    def __init__(self, markets: List[ExchangeAgent], sim_info, *, softlimit: int = 10, base_spread: float = 0.5, cash: float = 1_000):
        super().__init__(markets, cash)
        self.sim_info = sim_info
        self.soft     = max(1, softlimit)
        self.base     = max(base_spread, _MIN_PRICE)
        self.type     = 'FairPriceMarketMaker'

    def call(self) -> None:
        self.exercise_options()
        for mid, mkt in self.markets.items():
            opt = mkt.asset
            if not isinstance(opt, Option) or opt.is_expired():
                continue
            metrics = self.sim_info.option_metrics.get(opt.id)
            if not metrics or not metrics['price']:
                continue
            fair = metrics['price'][-1]
            if not _isfinite(fair):
                continue

            # cancel resting orders first
            for o in list(self.orders[mid]):
                self._cancel_order(mid, o)

            pos = self.assets.get(mid, 0)
            off = (pos / self.soft) * self.base
            bid = max(fair - self.base - off, _MIN_PRICE)
            ask = max(fair + self.base - off, _MIN_PRICE)
            qty_bid = max(1, self.soft - pos)
            qty_ask = max(1, self.soft + pos)
            self._buy_limit(mid, qty_bid, bid)
            self._sell_limit(mid, qty_ask, ask)

