"""
Extended multi‑asset trader implementations with additional
numerical‑stability safeguards.

Every potential division‑by‑zero, overflow or NaN/Inf propagator is now
wrapped with defensive checks so the agents can run for long horizons
without crashing or silently producing non‑finite values.
"""

from __future__ import annotations

import math
import random
from typing import List, Dict

from AgentBasedModel.traders import MultiTrader  # base multi‑asset trader
from AgentBasedModel.exchange import ExchangeAgent
from AgentBasedModel.utils import Order
from AgentBasedModel.utils.cost_basis import CostBasisFIFO

import numpy as np

# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

_MIN_PRICE = 1e-6
_EPS = 1e-8  # global tiny value to avoid zero‑division


def _is_finite(x: float) -> bool:
    """Convenience wrapper around math.isfinite."""
    return math.isfinite(x)


# ---------------------------------------------------------------------------
#  Base multi‑asset trader
# ---------------------------------------------------------------------------

class MultiAssetTrader(MultiTrader):
    """Base class for traders operating on multiple exchanges."""

    def __init__(self, markets: List[ExchangeAgent], cash: float = 1e3):
        super().__init__(markets, cash)
        # Per‑market positions and resting orders
        self.assets = {m.id: 0 for m in markets}
        self.orders = {m.id: [] for m in markets}
        self.type = "MultiAssetTrader"

        self._cb_trackers: Dict[int, CostBasisFIFO] = {}
        self.realized_pnl: float = 0.0

    # ............................................................... public –
    def equity(self) -> float:
        """Return cash + marked‑to‑market value of positions."""
        total = self.cash
        for mid, mkt in self.markets.items():
            price = mkt.price()
            if not _is_finite(price):
                continue  # skip broken quote
            total += self.assets.get(mid, 0) * price
        return total

    def income(self):
        """Apply dividends (if any) and risk‑free interest on cash."""
        for mid, mkt in self.markets.items():
            div = mkt.dividend()
            if _is_finite(div):
                self.cash += self.assets.get(mid, 0) * div
        rf = next(iter(self.markets.values())).risk_free_rate
        if _is_finite(rf):
            self.cash += self.cash * rf

    # ........................................................... order I/O –
    def _buy_limit(self, market_id, qty, price):
        if qty <= 0 or not _is_finite(price):
            return
        order = Order(round(price, 2), round(qty), "bid", self)
        self.orders[market_id].append(order)
        self.markets[market_id].limit_order(order)

    def _sell_limit(self, market_id, qty, price):
        if qty <= 0 or not _is_finite(price):
            return
        order = Order(round(price, 2), round(qty), "ask", self)
        self.orders[market_id].append(order)
        self.markets[market_id].limit_order(order)

    def _buy_market(self, market_id, qty) -> int:
        if qty <= 0:
            return 0
        book = self.markets[market_id].order_book["ask"]
        if not book:
            return qty
        price = book.first.price
        if not _is_finite(price):
            return qty
        order = Order(price, round(qty), "bid", self)
        return self.markets[market_id].market_order(order).qty

    def _sell_market(self, market_id, qty) -> int:
        if qty <= 0:
            return 0
        book = self.markets[market_id].order_book["bid"]
        if not book:
            return qty
        price = book.first.price
        if not _is_finite(price):
            return qty
        order = Order(price, round(qty), "ask", self)
        return self.markets[market_id].market_order(order).qty

    def _cancel_order(self, market_id, order: Order):
        if order in self.orders[market_id]:
            self.markets[market_id].cancel_order(order)
            self.orders[market_id].remove(order)


# ---------------------------------------------------------------------------
#  Random trader
# ---------------------------------------------------------------------------

class MultiAssetRandomTrader(MultiAssetTrader):
    """Noise trader spraying random orders across markets."""

    def __init__(self, markets: List[ExchangeAgent], cash: float = 1e3):
        super().__init__(markets, cash)
        self.type = "MultiAssetRandomTrader"

    # ~~~ price helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def draw_delta(std: float = 2.5):
        std = max(std, _EPS)
        return random.expovariate(1 / std)

    @staticmethod
    def draw_price(order_type: str, spread: dict, std: float = 2.5) -> float:
        if not spread:
            return float("nan")
        if random.random() < 0.35:  # inside spread
            return random.uniform(spread["bid"], spread["ask"])
        delta = MultiAssetRandomTrader.draw_delta(std)
        return spread["bid"] - delta if order_type == "bid" else spread["ask"] + delta

    # ~~~ main entry ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def call(self):
        if not self.markets:
            return
        mid = random.choice(list(self.markets.keys()))
        mkt = self.markets[mid]
        spread = mkt.spread()
        if not spread:
            return

        rnd = random.random()
        if rnd > 0.85:  # market order
            qty = random.randint(1, 5)
            (self._buy_market if random.random() > 0.5 else self._sell_market)(mid, qty)
        elif rnd > 0.5:  # limit order
            qty = random.randint(1, 5)
            side = "bid" if random.random() > 0.5 else "ask"
            price = self.draw_price(side, spread)
            if side == "bid":
                self._buy_limit(mid, qty, price)
            else:
                self._sell_limit(mid, qty, price)
        elif rnd < 0.35 and self.orders[mid]:  # cancel
            self._cancel_order(mid, random.choice(self.orders[mid]))


# ---------------------------------------------------------------------------
#  Fundamentalist trader
# ---------------------------------------------------------------------------

class MultiAssetFundamentalist(MultiAssetTrader):
    """Discounted‑dividend fundamentalist (uses 1st market as reference)."""

    def __init__(self, markets: List[ExchangeAgent], cash: float = 1e3, access: int = 0):
        super().__init__(markets, cash)
        self.access = access
        self.type = "MultiAssetFundamentalist"

    # ............................................................. helpers –
    @staticmethod
    def evaluate(divs: list, r: float) -> float:
        r = max(r, _EPS)
        perp = divs[-1] / r / (1 + r) ** (len(divs) - 1)
        known = (
            sum(divs[i] / (1 + r) ** (i + 1) for i in range(len(divs) - 1))
            if len(divs) > 1
            else 0
        )
        return known + perp

    @staticmethod
    def draw_quantity(pf: float, p: float, gamma: float = 5e-3) -> int:
        gamma = max(gamma, _EPS)
        if not (_is_finite(pf) and _is_finite(p) and p > 0):
            return 1
        q = abs(pf - p) / p / gamma
        return max(1, min(5, round(q)))

    # ................................................................. step –
    def call(self):
        if not self.markets:
            return
        mid = next(iter(self.markets))
        mkt = self.markets[mid]
        spread = mkt.spread()
        if not spread:
            return

        divs = mkt.dividend(self.access)
        pf = self.evaluate(divs, mkt.risk_free_rate)
        p = mkt.price()
        if not (_is_finite(pf) and _is_finite(p) and p > 0):
            return
        pf = round(pf, 4)

        tc = mkt.transaction_cost
        rnd = random.random()
        if rnd <= 0.45:  # cancel path
            if self.orders[mid]:
                self._cancel_order(mid, self.orders[mid][0])
            return

        rnd2 = random.random()
        ask_t = spread["ask"] * (1 + tc)
        bid_t = spread["bid"] * (1 - tc)
        qty = self.draw_quantity(pf, p)

        if pf >= ask_t:  # undervalued -> buy
            if rnd2 > 0.5:
                self._buy_market(mid, qty)
            else:
                price = (pf + MultiAssetRandomTrader.draw_delta()) * (1 + tc)
                self._sell_limit(mid, qty, price)
        elif pf <= bid_t:  # overvalued -> sell
            if rnd2 > 0.5:
                self._sell_market(mid, qty)
            else:
                price = (pf - MultiAssetRandomTrader.draw_delta()) * (1 - tc)
                self._buy_limit(mid, qty, price)
        else:  # inside band
            if rnd2 > 0.5:
                price = (pf - MultiAssetRandomTrader.draw_delta()) * (1 - tc)
                self._buy_limit(mid, qty, price)
            else:
                price = (pf + MultiAssetRandomTrader.draw_delta()) * (1 + tc)
                self._sell_limit(mid, qty, price)


# ---------------------------------------------------------------------------
#  Chartist / momentum trader
# ---------------------------------------------------------------------------

class MultiAssetChartist2D(MultiAssetTrader):
    """Sentiment‑driven momentum trader across markets."""

    def __init__(self, markets: List[ExchangeAgent], cash: float = 1_000):
        super().__init__(markets, cash)
        self.type = "MultiAssetChartist"
        self.sentiment = "Optimistic" if random.random() > 0.5 else "Pessimistic"

    # ............................................................. helpers –
    @staticmethod
    def _get_types(info):
        return [v[-1] for v in info.types.values() if v]

    @staticmethod
    def _get_sentiments(info):
        return [v[-1] for v in info.sentiments.values() if v]

    def change_sentiment(self, info, a1=1, a2=1, v1=0.1):
        if not self.markets:
            return
        n_traders = len(info.traders)
        types = self._get_types(info)
        sents = self._get_sentiments(info)
        n_chart = types.count("MultiAssetChartist")
        n_opt = sents.count("Optimistic")
        n_pes = sents.count("Pessimistic")

        mid = next(iter(self.markets))
        price = self.markets[mid].price()
        if price <= 0 or not _is_finite(price):
            return
        dp = 0.0
        if len(info.prices[mid]) > 1 and _is_finite(info.prices[mid][-2]):
            dp = info.prices[mid][-1] - info.prices[mid][-2]

        x = (n_opt - n_pes) / max(n_chart, 1)
        U = a1 * x + a2 / max(v1, _EPS) * dp / price
        U = max(-50, min(50, U))  # clamp to avoid overflow in exp

        if self.sentiment == "Optimistic":
            prob = v1 * n_chart / max(n_traders, 1) * math.exp(U)
            if random.random() < prob:
                self.sentiment = "Pessimistic"
        else:
            prob = v1 * n_chart / max(n_traders, 1) * math.exp(-U)
            if random.random() < prob:
                self.sentiment = "Optimistic"

    # ................................................................. step –
    def call(self):
        if not self.markets:
            return
        # choose market by best price given sentiment
        if self.sentiment == "Optimistic":
            mid = min(self.markets, key=lambda k: self.markets[k].price())
        else:
            mid = max(self.markets, key=lambda k: self.markets[k].price())
        mkt = self.markets[mid]
        spread = mkt.spread()
        if not spread:
            return

        rnd = random.random()
        tc = mkt.transaction_cost
        qty = random.randint(1, 5)

        if rnd > 0.85:  # market order
            if self.sentiment == "Optimistic":
                self._buy_market(mid, qty)
            else:
                self._sell_market(mid, qty)
        elif rnd > 0.5:  # limit order
            price = random.uniform(spread["bid"], spread["ask"])
            if self.sentiment == "Optimistic":
                self._buy_limit(mid, qty, price * (1 - tc))
            else:
                self._sell_limit(mid, qty, price * (1 + tc))
        elif self.orders[mid]:  # cancel
            self._cancel_order(mid, random.choice(self.orders[mid]))


# ---------------------------------------------------------------------------
#  Cross‑market market maker
# ---------------------------------------------------------------------------

class MultiAssetMarketMaker2D(MultiAssetTrader):
    """Inventory‑balanced market maker quoting two‑sided spreads."""

    def __init__(self, markets: List[ExchangeAgent], cash: float = 1e3, softlimit: int = 10):
        super().__init__(markets, cash)
        self.type = "MultiAssetMarketMaker2D"
        self.softlimit = max(1, softlimit)
        self.ul = self.softlimit
        self.ll = -self.softlimit
        self.panic = False

    # ................................................................. step –
    def call(self):
        for mid, mkt in self.markets.items():
            for o in list(self.orders[mid]):
                self._cancel_order(mid, o)

            spread = mkt.spread()
            if not spread:
                continue

            pos = self.assets.get(mid, 0)
            bid_v = max(0, self.ul - 1 - pos)
            ask_v = max(0, pos - self.ll - 1)
            center = (self.ul + self.ll) / 2

            # Panic logic: position out of range
            if bid_v == 0 and ask_v == 0:
                self.panic = True
                adjust = pos - center
                if adjust > 0:
                    self._sell_market(mid, adjust)
                elif adjust < 0:
                    self._buy_market(mid, -adjust)
                continue
            elif bid_v == 0:
                self.panic = True
                self._sell_market(mid, pos - center)
                continue
            elif ask_v == 0:
                self.panic = True
                self._buy_market(mid, center - pos)
                continue

            # Stable state: place symmetric quotes
            self.panic = False
            spr_width = spread["ask"] - spread["bid"]
            spr_width = max(spr_width, _EPS)
            offset = -(spr_width * (pos / self.softlimit))
            bid_price = spread["bid"] - offset - 0.1
            ask_price = spread["ask"] + offset + 0.1

            if _is_finite(bid_price):
                self._buy_limit(mid, bid_v, bid_price)
            if _is_finite(ask_price):
                self._sell_limit(mid, ask_v, ask_price)
