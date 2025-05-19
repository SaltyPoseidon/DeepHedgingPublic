"""
Order / OrderList — стакан с поддержкой:
• price-time priority
• FIFO-учёта реализованного PnL (CostBasisFIFO)
• корректного обновления cash **и** inventory (assets) для
  одноактивных и мультиактивных трейдеров.
"""

from __future__ import annotations
from typing import Any

from AgentBasedModel.utils.cost_basis import CostBasisFIFO

# ────────────────────────────────────────────────────────────── Order ──

class Order:
    order_id = 0

    def __init__(self, price, qty, order_type, trader_link=None):
        self.price      = price
        self.qty        = qty
        self.order_type = order_type            # 'bid' | 'ask'
        self.trader     = trader_link
        self.order_id   = Order.order_id
        Order.order_id += 1

        # doubly linked list pointers
        self.left  = None
        self.right = None

    # ------- price-time comparisons -----------------------------------
    def __lt__(self, other):
        if self.order_type != other.order_type:
            return (self.order_type == 'bid' and self.price > other.price) or \
                   (self.order_type == 'ask' and self.price < other.price)
        return (self.order_type == 'bid' and self.price > other.price) or \
               (self.order_type == 'ask' and self.price < other.price)

    def __le__(self, other):
        if self.order_type != other.order_type:
            return (self.order_type == 'bid' and self.price >= other.price) or \
                   (self.order_type == 'ask' and self.price <= other.price)
        return (self.order_type == 'bid' and self.price >= other.price) or \
               (self.order_type == 'ask' and self.price <= other.price)

    def __gt__(self, other): return not self.__le__(other)
    def __ge__(self, other): return not self.__lt__(other)

    def __repr__(self): return f"{self.order_type}(p={self.price}, q={self.qty})"

    # ------- (de)serialization ----------------------------------------
    def to_dict(self):
        return {'price': self.price,
                'qty': self.qty,
                'order_type': self.order_type,
                'trader_link': self.trader}

    @classmethod
    def from_dict(cls, d):
        return Order(d['price'], d['qty'], d['order_type'], d.get('trader_link'))

# ───────────────────────────────────────────────────────── iterator ──

class OrderIter:
    def __init__(self, order_list): self.order = order_list.first
    def __iter__(self): return self
    def __next__(self):
        if self.order:
            cur, self.order = self.order, self.order.right
            return cur
        raise StopIteration

# ─────────────────────────────────────────────────────── OrderList ──

class OrderList:
    """
    Двусвязный список одной стороны стакана.
    Поддерживает insert / append / push / remove / fulfill.
    """

    def __init__(self, order_type: str, market_id: Any = None):
        self.first = None
        self.last  = None
        self.order_type = order_type    # 'bid' | 'ask'
        self.market_id  = market_id     # нужен FIFO-учёту

    # ------------- container proto -----------------------------------
    def __iter__(self): return OrderIter(self)
    def __bool__(self): return self.first is not None
    def __len__(self):
        n = 0
        for _ in self: n += 1
        return n

    # ------------- low-level ops -------------------------------------
    def remove(self, order: Order):
        if order == self.first: self.first = order.right
        if order == self.last:  self.last  = order.left
        if order.left:  order.left.right  = order.right
        if order.right: order.right.left = order.left
        order.left = order.right = None

    def append(self, order: Order):
        if not self: self.first = self.last = order
        else:
            self.last.right = order
            order.left = self.last
            self.last = order

    def push(self, order: Order):
        if not self: self.first = self.last = order
        else:
            self.first.left = order
            order.right = self.first
            self.first = order

    def insert(self, order: Order):
        if not self: return self.append(order)
        if order <= self.first: return self.push(order)
        for cur in self:
            if order <= cur:
                order.left, order.right = cur.left, cur
                order.left.right = order
                cur.left = order
                return
        self.append(order)

    # ------------- matching engine -----------------------------------
    def fulfill(self, order: Order, t_cost: float) -> Order:
        if order.order_type == self.order_type:
            raise ValueError("fulfill(): sides must differ")

        # helper для FIFO-трекера + суммарного realized_pnl
        def _track(trader, side, qty, price):
            if trader is None: return
            cb = trader._cb_trackers.setdefault(self.market_id, CostBasisFIFO())
            (cb.buy if side == 'buy' else cb.sell)(qty, price)
            trader.realized_pnl = sum(c.realized_pnl
                                      for c in trader._cb_trackers.values())

        for book_order in self:
            if order.qty == 0 or book_order > order:
                break

            qty_exec = min(order.qty, book_order.qty)
            px       = book_order.price
            book_order.qty -= qty_exec
            order.qty      -= qty_exec

            # -------- settlement --------------------------------------
            if order.order_type == 'bid':           # order = buyer
                # ─── cash
                if order.trader:
                    order.trader.cash -= px * qty_exec * (1 + t_cost)
                if book_order.trader:
                    book_order.trader.cash += px * qty_exec * (1 - t_cost)
                # ─── inventory
                if order.trader:
                    if isinstance(order.trader.assets, dict):
                        order.trader.assets[self.market_id] = \
                            order.trader.assets.get(self.market_id, 0) + qty_exec
                    else:
                        order.trader.assets += qty_exec
                if book_order.trader:
                    if isinstance(book_order.trader.assets, dict):
                        book_order.trader.assets[self.market_id] = \
                            book_order.trader.assets.get(self.market_id, 0) - qty_exec
                    else:
                        book_order.trader.assets -= qty_exec
                # ─── FIFO
                _track(order.trader,      'buy',  qty_exec, px)
                _track(book_order.trader, 'sell', qty_exec, px)

            else:                                   # order = seller
                # ─── cash
                if order.trader:
                    order.trader.cash += px * qty_exec * (1 - t_cost)
                if book_order.trader:
                    book_order.trader.cash -= px * qty_exec * (1 + t_cost)
                # ─── inventory
                if order.trader:
                    if isinstance(order.trader.assets, dict):
                        order.trader.assets[self.market_id] = \
                            order.trader.assets.get(self.market_id, 0) - qty_exec
                    else:
                        order.trader.assets -= qty_exec
                if book_order.trader:
                    if isinstance(book_order.trader.assets, dict):
                        book_order.trader.assets[self.market_id] = \
                            book_order.trader.assets.get(self.market_id, 0) + qty_exec
                    else:
                        book_order.trader.assets += qty_exec
                # ─── FIFO
                _track(order.trader,      'sell', qty_exec, px)
                _track(book_order.trader, 'buy',  qty_exec, px)
            # ----------------------------------------------------------

            if book_order.qty == 0:
                self.remove(book_order)

        return order    # остаток (qty == 0 → исполнен полностью)

    # ------------- factory -------------------------------------------
    @classmethod
    def from_list(cls, data, *, sort=False, market_id=None):
        ol = cls(data[0]['order_type'], market_id)
        for d in data:
            (ol.insert if sort else ol.append)(Order.from_dict(d))
        return ol
