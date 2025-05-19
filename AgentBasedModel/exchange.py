"""
exchange.py – биржевые агенты (базовый + опционный) со встроенной
численной защитой из numerics.py.
"""

from __future__ import annotations
import random, math
from typing import Type, Union

import numpy as np

from AgentBasedModel.numerics import (
    clip_price, safe_div, safe_exp, finite, MIN_PRICE, EPS
)
from AgentBasedModel.utils import Order, OrderList
from AgentBasedModel.utils.math import exp                    # лог-норм-helpers


# ──────────────────────────────────────────────────────────────────────
#  БАЗОВЫЕ АКТИВЫ
# ──────────────────────────────────────────────────────────────────────
class Asset:
    id = 0
    def __init__(self):
        self.type = "Stock"
        self.name = f"Asset{Asset.id}"
        self.id   = Asset.id
        Asset.id += 1
    def update(self): ...


class Stock(Asset):
    def __init__(self, dividend: float | int):
        super().__init__()
        self.dividend: float = clip_price(dividend)
        self.dividend_book: list[float] = []
        self._fill_book()

    # ------------------------------------------------------------------
    @staticmethod
    def _next_dividend(cur: float, std: float = 5e-3) -> float:
        return clip_price(cur * max(exp(random.normalvariate(0, std)), 0.))

    def _fill_book(self, n: int = 100):
        if not self.dividend_book:
            self.dividend_book.append(self.dividend); n -= 1
        for _ in range(n):
            self.dividend_book.append(self._next_dividend(self.dividend_book[-1]))

    def update(self):
        self._fill_book(1)
        self.dividend = self.dividend_book.pop(0)


# ──────────────────────────────────────────────────────────────────────
#  EXCHANGE AGENT
# ──────────────────────────────────────────────────────────────────────
class ExchangeAgent:
    id = 0

    def __init__(
        self,
        asset: Type[Asset],
        risk_free_rate: float = 5e-4,
        transaction_cost: float = 0.0,
        mean: float | int = None,
        std: float | int = 25,
        n: int = 1000,
    ):
        if mean is None:
            mean = asset.dividend / max(risk_free_rate, EPS)

        self.name = f"Exchange{ExchangeAgent.id}"
        self.id   = ExchangeAgent.id
        ExchangeAgent.id += 1

        self.asset             = asset
        self.risk_free_rate    = risk_free_rate
        self.transaction_cost  = transaction_cost
        self.order_book        = {
            "bid": OrderList("bid", market_id=self.id),
            "ask": OrderList("ask", market_id=self.id),
        }
        self._fill_book(mean, std, n)

        self.last_spread = {
            'bid': self.order_book['bid'].first.price,
            'ask': self.order_book['ask'].first.price,
        }

    # ------------------------------------------------------------------
    def _fill_book(self, mean: float, std: float, n: int):
        prices_lo = [round(random.normalvariate(mean - std, std), 1) for _ in range(n // 2)]
        prices_hi = [round(random.normalvariate(mean + std, std), 1) for _ in range(n // 2)]
        quantities = [random.randint(1, 5) for _ in range(n)]

        for p, q in zip(sorted(prices_lo + prices_hi), quantities):
            side = "ask" if p > mean else "bid"
            order = Order(clip_price(p), q, side)
            if side == "ask":
                self.order_book["ask"].append(order)
            else:
                self.order_book["bid"].push(order)

    # ------------------------------------------------------------------
    def spread(self) -> dict:
        """Гарантируем ask ≥ bid + MIN_PRICE."""
        if not (self.order_book['bid'] and self.order_book['ask']):
            return self.last_spread                       # fallback
        bid = clip_price(self.order_book['bid'].first.price)
        ask = clip_price(self.order_book['ask'].first.price)
        if ask <= bid:                                    # enforce positive spread
            ask = bid + MIN_PRICE
        self.last_spread = {'bid': bid, 'ask': ask}
        return self.last_spread

    def price(self) -> float:
        s = self.spread()
        return finite((s['bid'] + s['ask']) * 0.5, default=MIN_PRICE)

    # ------------------------------------------------------------------
    def dividend(self, access: int | None = None):
        if self.asset.type != "Stock":
            return 0
        if access is None:
            return self.asset.dividend
        return [self.asset.dividend, *self.asset.dividend_book[:access]]

    # ------------------------------------------------------------------
    #  Приём ордеров (без изменений, кроме clip_price) …
    # ------------------------------------------------------------------
    def limit_order(self, order: Order):
        if not order.qty:
            return
        best = self.spread()
        tc   = self.transaction_cost

        if order.order_type == "bid":
            if order.price >= best["ask"]:
                order = self.order_book["ask"].fulfill(order, tc)
            if order.qty:
                self.order_book["bid"].insert(order)

        elif order.order_type == "ask":
            if order.price <= best["bid"]:
                order = self.order_book["bid"].fulfill(order, tc)
            if order.qty:
                self.order_book["ask"].insert(order)

    def market_order(self, order: Order) -> Order:
        tc = self.transaction_cost
        book = self.order_book["ask"] if order.order_type == "bid" else self.order_book["bid"]
        return book.fulfill(order, tc)

    def cancel_order(self, order: Order):
        self.order_book[order.order_type].remove(order)


# ──────────────────────────────────────────────────────────────────────
#  ОПЦИОННЫЙ EXCHANGE
# ──────────────────────────────────────────────────────────────────────
class Option(Asset):
    def __init__(self, underlying, strike, expiry, option_type, underlying_exchange):
        super().__init__()
        self.type      = "Option"
        self.underlying = underlying
        self.strike     = strike
        self.expiry     = expiry
        self.option_type = option_type.lower()
        self.underlying_exchange = underlying_exchange
        self.dividend = 0

    def update(self):
        if self.expiry > 0:
            self.expiry -= 1

    def is_expired(self): return self.expiry <= 0

    def exercise(self) -> Union[float, None]:
        """
        Compute option pay‑off using current best quote.

        Returns
        -------
        float | None
            Pay‑off (0 if out‑of‑the‑money) or *None*
            if the underlying has no quote at the moment.
        """
        spread = self.underlying_exchange.spread()
        if not spread:                        # no liquidity
            return None

        if self.option_type == "call":
            price = spread["bid"]
            return max(price - self.strike, 0.0)
        elif self.option_type == "put":
            price = spread["ask"]
            return max(self.strike - price, 0.0)
        raise ValueError("option_type must be 'call' or 'put'")

    # -------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Option(name={self.name}, strike={self.strike}, "
            f"expiry={self.expiry}, type={self.option_type})"
        )

class OptionExchangeAgent(ExchangeAgent):
    def __init__(
        self,
        underlying_exchange: ExchangeAgent,
        option: Option,
        *,
        sim_info,
        vol_window: int = 50,
        days_per_year: int = 250,
        risk_free_rate: float = 5e-4,
        transaction_cost: float = 0.0,
        std: float = 1.0,
        n: int = 1000,
    ):
        self.sim_info           = sim_info
        self.vol_window         = vol_window
        self.days_per_year      = days_per_year
        self.option             = option
        self.underlying_exchange= underlying_exchange
        super().__init__(option, risk_free_rate, transaction_cost,
                         mean=0.0, std=std, n=n)      # mean пересчитаем внутри
        self.name = f"OptionExchange{self.id}"

    # ------------------------------------------------------------------
    def _fill_book(self, _mean_unused, std: float, n: int):
        """Ориентируемся на BS-цену + случайный «шум»."""
        bs = self.sim_info.bs_option(
            self.option, idx_under=self.underlying_exchange.id,
            window=self.vol_window, days_in_year=self.days_per_year
        )
        fair = clip_price(bs.get("price", MIN_PRICE))

        half = n // 2
        # защищаем sigma_frac — иначе lognormvariate может взорваться
        sigma_frac = np.clip(safe_div(std, fair), 1e-4, 50)

        prices_bid, prices_ask = [], []
        for _ in range(half):
            try:
                p = random.lognormvariate(
                    math.log(fair) - 0.5 * sigma_frac**2,
                    sigma_frac
                )
            except (ValueError, OverflowError):
                p = fair
            prices_bid.append(clip_price(p))

        for _ in range(half):
            p = random.normalvariate(fair + std, std)
            prices_ask.append(clip_price(p))

        quantities = [random.randint(1, 5) for _ in range(n)]
        self.order_book = {
            "bid": OrderList("bid", market_id=self.id),
            "ask": OrderList("ask", market_id=self.id),
        }
        for p, q in zip(sorted(prices_bid + prices_ask), quantities):
            side = "ask" if p > fair else "bid"
            order = Order(p, q, side)
            if side == "ask":
                self.order_book["ask"].append(order)
            else:
                self.order_book["bid"].push(order)

        # «фиктивные» уровни, если нужно
        if not self.order_book['bid']:
            self.order_book['bid'].push(Order(fair * 0.9, 1, 'bid'))
        if not self.order_book['ask']:
            self.order_book['ask'].append(Order(fair * 1.1, 1, 'ask'))

        self.last_spread = {
            'bid': self.order_book['bid'].first.price,
            'ask': self.order_book['ask'].first.price,
        }

    def price(self) -> float:
        s = self.spread()
        return finite((s['bid'] + s['ask']) * 0.5, default=MIN_PRICE)
