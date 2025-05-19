from typing import List

from AgentBasedModel.exchange import ExchangeAgent
from AgentBasedModel.utils import Order
from AgentBasedModel.utils.math import exp, mean
import random

from AgentBasedModel.utils.cost_basis import CostBasisFIFO

class Trader:
    id = 0

    def __init__(
            self,
            cash:   float | int = 10**3,
            assets: int = 0
        ):
        """Trader buys and sells Asset through ExchangeAgent

        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares hold, defaults 0 
        """
        self.type = 'Unknown'
        self.name = f'Trader{self.id}'
        self.id = Trader.id
        Trader.id += 1

        self.cash = cash
        self.assets = assets

        # ► FIFO-трекинг по market_id
        self._cb_trackers: dict[int, CostBasisFIFO] = {}
        self.realized_pnl: float = 0.0      # кумулятивный

    def __str__(self) -> str:
        return f'{self.name} ({self.type})'

    def __repr__(self) -> str:
        return f'{self.name}'


class SingleTrader(Trader):
    def __init__(
            self,
            market: ExchangeAgent,
            cash:   float | int = 10**3,
            assets: int = 0
        ):
        """SingleTrader trades on a single exchange

        :param market: link to exchange agent
        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares, defaults 0 
        """
        super().__init__(cash, assets)
        self.market = market
        self.orders = list()

    def equity(self):
        price = self.market.price()
        return self.cash + self.assets * price

    def income(self):
        self.cash += self.assets * self.market.dividend()      # Dividend payments
        self.cash += self.cash   * self.market.risk_free_rate  # Interest payment

    def _buy_limit(self, quantity, price):
        order = Order(round(price, 1), round(quantity), 'bid', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _sell_limit(self, quantity, price):
        order = Order(round(price, 1), round(quantity), 'ask', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _buy_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.market.order_book['ask']:
            return quantity
        
        price = self.market.order_book['ask'].last.price
        order = Order(price, round(quantity), 'bid', self)
        return self.market.market_order(order).qty

    def _sell_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.market.order_book['bid']:
            return quantity
        
        price = self.market.order_book['bid'].last.price
        order = Order(price, round(quantity), 'ask', self)
        return self.market.market_order(order).qty

    def _cancel_order(self, order: Order):
        self.market.cancel_order(order)
        self.orders.remove(order)


class MultiTrader(Trader):
    def __init__(
            self,
            markets: List[ExchangeAgent],
            cash:    float | int = 10**3,
            assets:  int = 0
        ):
        """MultiTrader trades on a multiple exchanges

        :param markets: links to exchange agents
        :param cash: trader's cash available, defaults to 10**3
        :param assets: trader's number of shares of each type (key: asset.id), defaults to None
        """
        super().__init__(cash, assets)
        self.markets = {market.id: market for market in markets}
        self.orders  = {market.id: list() for market in markets}

    def equity(self):
        price = max([market.price() for market in self.markets.values()])
        return self.cash + self.assets * price

    def income(self):
        market = list(self.markets.values())[0]           # for now only single stock type allowed
        self.cash += self.assets * market.dividend()      # Dividend payments
        self.cash += self.cash   * market.risk_free_rate  # Interest payment

    def _buy_limit(self, market_id, quantity, price):
        order = Order(round(price, 1), round(quantity), 'bid', self)
        self.orders[market_id].append(order)
        self.markets[market_id].limit_order(order)

    def _sell_limit(self, market_id, quantity, price):
        order = Order(round(price, 1), round(quantity), 'ask', self)
        self.orders[market_id].append(order)
        self.markets[market_id].limit_order(order)

    def _buy_market(self, market_id, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.markets[market_id].order_book['ask']:
            return quantity
        
        price = self.markets[market_id].order_book['ask'].last.price
        order = Order(price, round(quantity), 'bid', self)
        return self.markets[market_id].market_order(order).qty

    def _sell_market(self, market_id, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.markets[market_id].order_book['bid']:
            return quantity
        
        price = self.markets[market_id].order_book['bid'].last.price
        order = Order(price, round(quantity), 'ask', self)
        return self.markets[market_id].market_order(order).qty

    def _cancel_order(self, market_id, order: Order):
        self.markets[market_id].cancel_order(order)
        self.orders[market_id].remove(order)


class Random(SingleTrader):
    def __init__(
            self,
            market: ExchangeAgent,
            cash:   float | int = 10**3,
            assets: int = 0
        ):
        """Random creates noisy orders to recreate trading in real environment

        :param market: link to exchange agent
        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares hold, defaults 0 
        """
        super().__init__(market, cash, assets)
        self.type = 'Random'

    @staticmethod
    def draw_delta(std: float | int = 2.5):
        lamb = 1 / std
        return random.expovariate(lamb)

    @staticmethod
    def draw_price(order_type, spread: dict, std: float | int = 2.5) -> float:
        """Draw price for limit order. The price is calculated as:

        - 0.35 probability: within the spread - uniform distribution
        - 0.65 probability: out of the spread - delta from best price is exponential distribution r.v.
        :return: order price
        """
        random_state = random.random()  # Determines IN spread OR OUT of spread

        # Within the spread
        if random_state < .35:
            return random.uniform(spread['bid'], spread['ask'])

        # Out of spread
        else:
            delta = Random.draw_delta(std)
            if order_type == 'bid':
                return spread['bid'] - delta
            if order_type == 'ask':
                return spread['ask'] + delta

    @staticmethod
    def draw_quantity(a=1, b=5) -> float:
        """Draw order quantity for limit or market order. The quantity is drawn from U[a,b]

        :param a: minimal quantity
        :param b: maximal quantity
        :return: order quantity
        """
        return random.randint(a, b)

    def call(self):
        spread = self.market.spread()
        if spread is None:
            return

        random_state = random.random()

        if random_state > .5:
            order_type = 'bid'
        else:
            order_type = 'ask'

        random_state = random.random()
        # Market order
        if random_state > .85:
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_market(quantity)
            elif order_type == 'ask':
                self._sell_market(quantity)

        # Limit order
        elif random_state > .5:
            price = self.draw_price(order_type, spread)
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_limit(quantity, price)
            elif order_type == 'ask':
                self._sell_limit(quantity, price)

        # Cancellation order
        elif random_state < .35:
            if self.orders:
                order_n = random.randint(0, len(self.orders) - 1)
                self._cancel_order(self.orders[order_n])


class Fundamentalist(SingleTrader):
    def __init__(
            self,
            market: ExchangeAgent,
            cash:   float | int = 10**3,
            assets: int = 0,
            access: int = 0
        ):
        """Fundamentalist trades based on asset fundamental value

        :param market: link to exchange agent
        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares hold, defaults 0
        :param access: number of future dividends known, defaults 0 (knows only current dividend)
        """
        super().__init__(market, cash, assets)
        self.type = 'Fundamentalist'
        self.access = access

    @staticmethod
    def evaluate(dividends: list, risk_free_rate: float):
        """Evaluates the stock using Constant Dividend Model.

        We first sum all known (based on *access*) discounted dividends. Then calculate perpetual value
        of the stock based on last known dividend.
        """
        divs = dividends  # known future dividends
        r = risk_free_rate

        perp = divs[-1] / r / (1 + r)**(len(divs) - 1)  # perpetual payments
        known = sum([divs[i] / (1 + r)**(i + 1) for i in range(len(divs) - 1)]) if len(divs) > 1 else 0
        return known + perp

    @staticmethod
    def draw_quantity(pf, p, gamma: float = 5e-3):
        """
        Draw order quantity for limit or market order. The quantity depends on difference between fundamental
        value and market price.

        :param pf: fundamental value
        :param p: market price
        :param gamma: dependency coefficient
        :return: order quantity
        """
        q = round(abs(pf - p) / p / gamma)
        return min(q, 5)

    def call(self):
        divs = self.market.dividend(self.access)
        rf = self.market.risk_free_rate

        pf = round(self.evaluate(divs, rf), 1)  # fundamental price
        p = self.market.price()
        spread = self.market.spread()
        t_cost = self.market.transaction_cost
        
        if spread is None:
            return

        random_state = random.random()

        # Limit or Market order
        if random_state > .45:
            random_state = random.random()

            ask_t = round(spread['ask'] * (1 + t_cost), 1)
            bid_t = round(spread['bid'] * (1 - t_cost), 1)

            if pf >= ask_t:
                if random_state > .5:
                    self._buy_market(
                        Fundamentalist.draw_quantity(pf, p)
                    )
                else:
                    self._sell_limit(
                        Fundamentalist.draw_quantity(pf, p),
                        (pf + Random.draw_delta()) * (1 + t_cost)
                    )

            elif pf <= bid_t:
                if random_state > .5:
                    self._sell_market(
                        Fundamentalist.draw_quantity(pf, p)
                    )
                else:
                    self._buy_limit(
                        Fundamentalist.draw_quantity(pf, p),
                        (pf - Random.draw_delta()) * (1 - t_cost)
                    )

            elif ask_t > pf > bid_t:
                if random_state > .5:
                    self._buy_limit(
                        Fundamentalist.draw_quantity(pf, p),
                        (pf - Random.draw_delta()) * (1 - t_cost)
                    )
                else:
                    self._sell_limit(
                        Fundamentalist.draw_quantity(pf, p),
                        (pf + Random.draw_delta()) * (1 + t_cost)
                    )

        # Cancel order
        else:
            if self.orders:
                self._cancel_order(self.orders[0])


class Chartist1D(SingleTrader):
    """
    Chartists are searching for trends in the price movements. Each trader has sentiment - opinion
    about future price movement (either increasing, or decreasing). Based on sentiment trader either
    buys stock or sells. Sentiment revaluation happens at the end of each iteration based on opinion
    propagation among other chartists, current price changes
    """
    def __init__(
            self,
            market: ExchangeAgent,
            cash:   float | int = 10**3,
            assets: int = 0
        ):
        """Chartist1D trades on single exchange based on sentiment

        :param market: link to exchange agent
        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares, defaults 0 
        """
        super().__init__(market, cash, assets)
        self.type = 'Chartist'
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'

    def call(self):
        random_state = random.random()
        t_cost = self.market.transaction_cost
        spread = self.market.spread()

        if self.sentiment == 'Optimistic':
            # Market order
            if random_state > .85:
                self._buy_market(
                    Random.draw_quantity()
                )
            # Limit order
            elif random_state > .5:
                self._buy_limit(
                    Random.draw_quantity(),
                    Random.draw_price('bid', spread) * (1 - t_cost)
                )
            # Cancel order
            elif random_state < .35:
                if self.orders:
                    self._cancel_order(self.orders[-1])

        elif self.sentiment == 'Pessimistic':
            # Market order
            if random_state > .85:
                self._sell_market(
                    Random.draw_quantity()
                )
            # Limit order
            elif random_state > .5:
                self._sell_limit(
                    Random.draw_quantity(),
                    Random.draw_price('ask', spread) * (1 + t_cost)
                )
            # Cancel order
            elif random_state < .35:
                if self.orders:
                    self._cancel_order(self.orders[-1])

    def _get_types(self, info):
        res = list()
        for v in info.types.values():
            res.append(v[-1] if v else None)
        return res

    def _get_sentiments(self, info):
        res = list()
        for v in info.sentiments.values():
            res.append(v[-1] if v else None)
        return res

    def change_sentiment(self, info, a1=1, a2=1, v1=.1):
        """
        Revaluate chartist's opinion about future price movement

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param v1: frequency of revaluation of opinion for sentiment
        """
        n_traders    = len(info.traders)  # number of all traders
        n_chartists  = sum([tr_type == 'Chartist'    for tr_type in self._get_types(info)])
        n_optimistic = sum([tr_type == 'Optimistic'  for tr_type in self._get_sentiments(info)])
        n_pessimists = sum([tr_type == 'Pessimistic' for tr_type in self._get_sentiments(info)])

        dp = (
            info.prices[self.market.id][-1] - info.prices[self.market.id][-2]  # price derivative
            if len(info.prices[self.market.id]) > 1 else 0
        )
        p = self.market.price()                                                # market price
        x = (n_optimistic - n_pessimists) / n_chartists

        U = a1 * x + a2 / v1 * dp / p
        if self.sentiment == 'Optimistic':
            prob = v1 * n_chartists / n_traders * exp(U)
            if prob > random.random():
                self.sentiment = 'Pessimistic'

        elif self.sentiment == 'Pessimistic':
            prob = v1 * n_chartists / n_traders * exp(-U)
            if prob > random.random():
                self.sentiment = 'Optimistic'


class Chartist2D(MultiTrader):
    """
    Similar to Chartist1D but can trade on multiple exchanges
    """
    def __init__(
            self,
            markets: List[ExchangeAgent],
            cash:    float | int = 10**3,
            assets:  int = 0
        ):
        """Chartist2D trades on multiple exchanges based on sentiment

        :param market: links to exchange agents
        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares, defaults 0 
        """
        super().__init__(markets, cash, assets)
        self.type = 'Chartist'
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'

    def call(self):
        f = min if self.sentiment == 'Optimistic' else max                # agg based on sentiment
        idx = f(self.markets, key=lambda idx: self.markets[idx].price())  # market with best price

        random_state = random.random()
        t_cost = self.markets[idx].transaction_cost
        spread = self.markets[idx].spread()

        if self.sentiment == 'Optimistic':
            # Market order
            if random_state > .85:
                self._buy_market(
                    idx,
                    Random.draw_quantity()
                )
            # Limit order
            if random_state > .5:
                self._buy_limit(
                    idx,
                    Random.draw_quantity(),
                    Random.draw_price('bid', spread) * (1 - t_cost)
                )
            # Cancel order
            elif random_state < .35:
                if self.orders[idx]:
                    self._cancel_order(idx, self.orders[idx][-1])

        elif self.sentiment == 'Pessimistic':
            # Market order
            if random_state > .85:
                self._sell_market(
                    idx,
                    Random.draw_quantity()
                )
            # Limit order
            if random_state > .5:
                self._sell_limit(
                    idx,
                    Random.draw_quantity(),
                    Random.draw_price('ask', spread) * (1 + t_cost)
                )
            # Cancel order
            elif random_state < .35:
                if self.orders[idx]:
                    self._cancel_order(idx, self.orders[idx][-1])

    def _get_types(self, info):
        res = list()
        for v in info.types.values():
            res.append(v[-1] if v else None)
        return res

    def _get_sentiments(self, info):
        res = list()
        for v in info.sentiments.values():
            res.append(v[-1] if v else None)
        return res

    def change_sentiment(self, info, a1=1, a2=1, v1=.1):
        """
        Revaluate chartist's opinion about future price movement

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param v1: frequency of revaluation of opinion for sentiment
        """
        n_traders    = len(info.traders)  # number of all traders
        n_chartists  = sum([tr_type == 'Chartist'    for tr_type in self._get_types(info)])
        n_optimistic = sum([tr_type == 'Optimistic'  for tr_type in self._get_sentiments(info)])
        n_pessimists = sum([tr_type == 'Pessimistic' for tr_type in self._get_sentiments(info)])

        for idx in self.markets.keys():
            dp = (
                info.prices[idx][-1] - info.prices[idx][-2]  # price derivative
                if len(info.prices[idx]) > 1 else 0
            )
            p = self.markets[idx].price()                    # market price
            x = (n_optimistic - n_pessimists) / n_chartists

            U = a1 * x + a2 / v1 * dp / p
            if self.sentiment == 'Optimistic':
                prob = v1 * n_chartists / n_traders * exp(U)
                if prob > random.random():
                    self.sentiment = 'Pessimistic'

            elif self.sentiment == 'Pessimistic':
                prob = v1 * n_chartists / n_traders * exp(-U)
                if prob > random.random():
                    self.sentiment = 'Optimistic'


class Universalist(Fundamentalist, Chartist1D):
    def __init__(
            self,
            market: ExchangeAgent,
            cash:   float | int = 10**3,
            assets: int = 0,
            access: int = 0
        ):
        """Universalist chooses between Fundamentalist and Chartist1D strategies

        :param market: link to exchange agent
        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares hold, defaults 0
        :param access: number of future dividends known, defaults 0 (knows only current dividend)
        """
        super().__init__(market, cash, assets)
        self.type      = 'Chartist'   if random.random() > .5 else 'Fundamentalist'  # randomly decide type
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'     # randomly decide sentiment (Chartist)
        self.access = access                                                         # future dividends known (Fundamentalist)

    def _get_types(self, info):
        res = list()
        for v in info.types.values():
            res.append(v[-1] if v else None)
        return res

    def _get_sentiments(self, info):
        res = list()
        for v in info.sentiments.values():
            res.append(v[-1] if v else None)
        return res

    def _get_returns(self, info):
        res = list()
        for v in info.returns.values():
            res.append(v[-1] if v else 0)
        return res

    def call(self):
        """
        Call one of parents' methods depending on what type it is currently set.
        """
        if self.type == 'Chartist':
            Chartist1D.call(self)
        elif self.type == 'Fundamentalist':
            Fundamentalist.call(self)

    def change_strategy(self, info, a1=1, a2=1, a3=1, v1=.1, v2=.1, s=.1):
        """
        Change strategy or sentiment

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param a3: importance of fundamentalist profit
        :param v1: frequency of revaluation of opinion for sentiment
        :param v2: frequency of revaluation of opinion for strategy
        :param s: importance of fundamental value opportunities
        """
        # Gather variables
        n_traders         = len(info.traders)  # number of all traders
        n_fundamentalists = sum([tr_type == 'Fundamentalist' for tr_type in self._get_types(info)])
        n_optimistic      = sum([tr_type == 'Optimistic'     for tr_type in self._get_sentiments(info)])
        n_pessimists      = sum([tr_type == 'Pessimistic'    for tr_type in self._get_sentiments(info)])

        dp = (
            info.prices[self.market.id][-1] - info.prices[self.market.id][-2]              # price derivative
            if len(info.prices[self.market.id]) > 1 else 0
        )
        p = self.market.price()                                                            # market price
        pf = self.evaluate(self.market.dividend(self.access), self.market.risk_free_rate)  # fundamental price

        r = pf * self.market.risk_free_rate  # expected dividend return
        R = mean(self._get_returns(info))    # average return in economy

        # Change sentiment
        if self.type == 'Chartist':
            Chartist1D.change_sentiment(self, info, a1, a2, v1)

        # Change strategy
        U1 = max(-100, min(100, a3 * ((r + 1 / v2 * dp) / p - R - s * abs((pf - p) / p))))
        U2 = max(-100, min(100, a3 * (R - (r + 1 / v2 * dp) / p - s * abs((pf - p) / p))))

        if self.type == 'Chartist':
            if self.sentiment == 'Optimistic':
                prob = v2 * n_optimistic / (n_traders * exp(U1))
                if prob > random.random():
                    self.type = 'Fundamentalist'
            elif self.sentiment == 'Pessimistic':
                prob = v2 * n_pessimists / (n_traders * exp(U2))
                if prob > random.random():
                    self.type = 'Fundamentalist'

        elif self.type == 'Fundamentalist':
            prob = v2 * n_fundamentalists / (n_traders * exp(-U1))
            if prob > random.random() and self.sentiment == 'Pessimistic':
                self.type = 'Chartist'
                self.sentiment = 'Optimistic'

            prob = v2 * n_fundamentalists / (n_traders * exp(-U2))
            if prob > random.random() and self.sentiment == 'Optimistic':
                self.type = 'Chartist'
                self.sentiment = 'Pessimistic'


class MarketMaker1D(SingleTrader):
    def __init__(
            self,
            market:    ExchangeAgent,
            cash:      float | int = 10**3,
            assets:    int = 0,
            softlimit: int = 100
        ):
        """MarketMaker1D (single exchange) creates limit orders on both sides of the glass

        :param market: link to exchange agent
        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares, defaults 0
        :param softlimit: determines upper and lower limits, defaults 100
        """
        super().__init__(market, cash, assets)
        self.type = 'Market Maker'
        self.softlimit = softlimit

        self.ul = softlimit   # assets lower limit
        self.ll = -softlimit  # assets upper limit
        self.panic = False    # if in panic -> use market orders

    def call(self):
        # Clear previous orders
        for order in self.orders.copy():
            self._cancel_order(order)

        spread = self.market.spread()

        # Calculate bid and ask volume
        bid_volume = max(0., self.ul - 1 - self.assets)
        ask_volume = max(0., self.assets - self.ll - 1)

        # If in panic state we use market orders
        if not (bid_volume or ask_volume):
            self.panic = True
            self._buy_market((self.ul + self.ll) / 2 - self.assets)  if ask_volume is None else None
            self._sell_market(self.assets - (self.ul + self.ll) / 2) if bid_volume is None else None
        
        # Otherwise use limit orders
        else:
            self.panic = False
            base_offset = -((spread['ask'] - spread['bid']) * (self.assets / self.softlimit))  # Price offset
            self._buy_limit(bid_volume, spread['bid'] - base_offset - .1)                      # BID
            self._sell_limit(ask_volume, spread['ask'] + base_offset + .1)                     # ASK


class MarketMaker2D(MultiTrader):
    def __init__(
            self,
            markets:   List[ExchangeAgent],
            cash:      float | int = 10**3,
            assets:    int = 0,
            softlimit: int = 10
        ):
        """MarketMaker2D (multi exchange) creates limit orders on both sides of the glass

        :param markets: links to exchange agents
        :param cash: trader's cash available, defaults 10**3
        :param assets: trader's number of shares for each exchange, defaults 0
        :param softlimit: determines upper and lower limits for each exchange, defaults 100
        """
        super().__init__(markets, cash, assets)
        self.type = 'Market Maker'
        self.softlimit = softlimit

        self.ul = softlimit   # assets lower limit
        self.ll = -softlimit  # assets upper limit
        self.panic = False    # if in panic -> use market orders

    def call(self):
        # Clear previous orders
        for idx, market in self.markets.items():
            for order in self.orders[idx].copy():
                self._cancel_order(idx, order)
        
        # Calculate bid and ask volume
        bid_volume = max(0., self.ul - 1 - self.assets)
        ask_volume = max(0., self.assets - self.ll - 1)

        # Panic state
        if not (bid_volume and ask_volume):
            self.panic = True
            if not bid_volume:
                idx = max(self.markets, key=lambda idx: self.markets[idx].price())  # market with the best price
                self._sell_market(idx, self.assets - (self.ul + self.ll) / 2)
            elif not ask_volume:
                idx = min(self.markets, key=lambda idx: self.markets[idx].price())  # market with the best price
                self._buy_market(idx, (self.ul + self.ll) / 2 - self.assets)
        
        # Stable state
        else:
            for idx, market in self.markets.items():
                self.panic = False
                spread = market.spread()
                base_offset = -((spread['ask'] - spread['bid']) * (self.assets / self.softlimit))  # Price offset
                self._buy_limit(idx, bid_volume, spread['bid'] - base_offset - .1)                 # BID
                self._sell_limit(idx, ask_volume, spread['ask'] + base_offset + .1)                # ASK
