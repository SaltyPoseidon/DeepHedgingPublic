from AgentBasedModel.simulator import SimulatorInfo
import AgentBasedModel.utils.math as math

import matplotlib.pyplot as plt
import numpy as np


def plot_price(
        info:    SimulatorInfo,
        idx:     int   = None,
        spread:  bool  = False,
        rolling: int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot stock market price

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id, defaults to None (all exchanges)
    :param spread: bid-ask prices, defaults to False (only when idx is not None)
    :param rolling: MA applied to list, defaults to 1
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title('Stock Market price') if rolling == 1 else plt.title(f'Stock Price (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    # plot 1 exchange
    if idx is not None:
        exchange = info.exchanges[idx]
        values = math.rolling(info.prices[idx], rolling)
        iterations = range(rolling - 1, len(values) + rolling - 1)
        
        plt.plot(iterations, values, color='black', label=exchange.name)

        if spread:
            v1 = math.rolling([el['bid'] for el in info.spreads[idx]], rolling)
            v2 = math.rolling([el['ask'] for el in info.spreads[idx]], rolling)

            plt.plot(iterations, v1, label='bid', color='green')
            plt.plot(iterations, v2, label='ask', color='red')

    # plot N exchanges
    else:
        for k, v in info.exchanges.items():
            values = math.rolling(info.prices[k], rolling)
            iterations = range(rolling - 1, len(values) + rolling - 1)

            plt.plot(iterations, values, label=v.name)

    plt.legend()
    plt.show()


def plot_price_fundamental(
        info:    SimulatorInfo,
        idx:     int   = None,
        access:  int   = 0,
        rolling: int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot stock market price and fundamental price (single asset)

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id, defaults to None (all exchanges)
    :parap access: Fundamentalist's number of known dividends, defaults to 0
    :param rolling: MA applied to list, defaults to 1
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title(
              'Stock Market and Fundamental price' if rolling == 1
        else f'Stock Market and Fundamental price (MA {rolling})'
    )
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    # plot 1 exchange
    if idx is not None:
        exchange = info.exchanges[idx]
        m_values = math.rolling(info.prices[idx], rolling)                     # market prices
        f_values = math.rolling(info.fundamental_value(idx, access), rolling)  # fundamental prices
        iterations = range(rolling - 1, len(m_values) + rolling - 1)

        plt.plot(iterations, m_values, color='tab:blue', alpha=1,  ls='-',  label=f'{exchange.name}: market')
        plt.plot(iterations, f_values, color='black',    alpha=.6, ls='--', label=f'{exchange.name}: fundamental')
    
    # plot N exchanges
    else:
        for k, v in info.exchanges.items():
            m_values = math.rolling(info.prices[k], rolling)                   # market prices
            iterations = range(rolling - 1, len(m_values) + rolling - 1)
            plt.plot(iterations, m_values, ls='-', label=f'{v.name}: market')

        f_values = math.rolling(info.fundamental_value(0, access), rolling)    # fundamental prices
        plt.plot(iterations, f_values, color='black', alpha=.6, ls='--', label=f'fundamental')

    plt.legend()
    plt.show()


def plot_price_fundamental_m(
        info:    SimulatorInfo,
        idx:     int   = None,
        access:  int   = 0,
        rolling: int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot stock market price and fundamental price (multiple assets)

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id, defaults to None (all exchanges)
    :parap access: Fundamentalist's number of known dividends, defaults to 0
    :param rolling: MA applied to list, defaults to 1
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title(
              'Stock Market and Fundamental price' if rolling == 1
        else f'Stock Market and Fundamental price (MA {rolling})'
    )
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    # plot 1 exchange
    if idx is not None:
        exchange = info.exchanges[idx]
        m_values = math.rolling(info.prices[idx], rolling)                     # market prices
        f_values = math.rolling(info.fundamental_value(idx, access), rolling)  # fundamental prices
        iterations = range(rolling - 1, len(m_values) + rolling - 1)

        plt.plot(iterations, m_values, color='tab:blue', alpha=1, ls='-',   label=f'{exchange.name}: market')
        plt.plot(iterations, f_values, color='tab:blue', alpha=.4, ls='--', label=f'{exchange.name}: fundamental')
    
    # plot N exchanges
    else:
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(info.exchanges))))
        
        for k, v in info.exchanges.items():
            m_values = math.rolling(info.prices[k], rolling)                     # market prices
            f_values = math.rolling(info.fundamental_value(k, access), rolling)  # fundamental prices
            iterations = range(rolling - 1, len(m_values) + rolling - 1)
            
            c = next(colors)
            plt.plot(iterations, m_values, color=c, alpha=1, ls='-',   label=f'{v.name}: market')
            plt.plot(iterations, f_values, color=c, alpha=.4, ls='--', label=f'{v.name}: fundamental')

    plt.legend()
    plt.show()


def plot_arbitrage(
        info:    SimulatorInfo,
        idx:     int   = None,
        access:  int   = 0,
        rolling: int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot % difference between stock market price and fundamental price

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id, defaults to None (all exchanges)
    :parap access: Fundamentalist's number of known dividends, defaults to 0
    :param rolling: MA applied to list, defaults to 1
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title(
              'Stock Market and Fundamental price difference %' if rolling == 1
        else f'Stock Market and Fundamental price difference % (MA {rolling})'
    )
    plt.xlabel('Iterations')
    plt.ylabel('Difference %')

    # plot 1 exchange
    if idx is not None:
        exchange = info.exchanges[idx]
        m_values = math.rolling(info.prices[idx], rolling)                     # market prices
        f_values = math.rolling(info.fundamental_value(idx, access), rolling)  # fundamental prices
        values = [
            100 * (m_values[i] - f_values[i]) / m_values[i]                    # arbitrage opportunity %
            for i in range(len(m_values))
        ]
        iterations = range(rolling - 1, len(values) + rolling - 1)

        plt.plot(iterations, values, color='black', label=exchange.name)
    
    # plot N exchanges
    else:
        for k, v in info.exchanges.items():
            m_values = math.rolling(info.prices[k], rolling)                     # market prices
            f_values = math.rolling(info.fundamental_value(k, access), rolling)  # fundamental prices
            values = [
                100 * (m_values[i] - f_values[i]) / m_values[i]                  # arbitrage opportunity %
                for i in range(len(m_values))
            ]
            iterations = range(rolling - 1, len(values) + rolling - 1)
            
            plt.plot(iterations, values, label=v.name)

    plt.legend()
    plt.show()


def plot_dividend(
        info:    SimulatorInfo,
        idx:     int   = None,
        rolling: int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot stock dividend

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id, defaults to None (all exchanges)
    :param rolling: MA applied to list, defaults to 1
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title('Stock Dividend') if rolling == 1 else plt.title(f'Stock Dividend (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Dividend')

    # plot 1 exchange
    if idx is not None:
        exchange = info.exchanges[idx]
        values = math.rolling(info.dividends[idx], rolling)
        iterations = range(rolling - 1, len(values) + rolling - 1)
        
        plt.plot(iterations, values, color='black', label=exchange.name)

    # plot N exchanges
    else:
        for k, v in info.exchanges.items():
            values = math.rolling(info.dividends[k], rolling)
            iterations = range(rolling - 1, len(values) + rolling - 1)

            plt.plot(iterations, values, label=v.name)

    plt.legend()
    plt.show()


def plot_liquidity(
        info:    SimulatorInfo,
        idx:     int   = None,
        rolling: int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot stock liquidity

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id, defaults to None (all exchanges)
    :param rolling: MA applied to list, defaults to 1
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title('Stock Liquidity') if rolling == 1 else plt.title(f'Stock Liquidity (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Liquidity')

    # plot 1 exchange
    if idx is not None:
        exchange = info.exchanges[idx]
        values = math.rolling(info.liquidity(idx), rolling)
        iterations = range(rolling - 1, len(values) + rolling - 1)
        
        plt.plot(iterations, values, color='black', label=exchange.name)

    # plot N exchanges
    else:
        for k, v in info.exchanges.items():
            values = math.rolling(info.liquidity(k), rolling)
            iterations = range(rolling - 1, len(values) + rolling - 1)

            plt.plot(iterations, values, label=v.name)

    plt.legend()
    plt.show()


def plot_volatility_price(
        info:    SimulatorInfo,
        idx:     int   = None,
        window:  int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot stock price volatility

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id, defaults to None (all exchanges)
    :param window: sample size to calculate std, > 1, defaults to 5
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title(f'Stock Price Volatility (WINDOW: {window})')
    plt.xlabel('Iterations')
    plt.ylabel('Price Volatility')

    # plot 1 exchange
    if idx is not None:
        exchange = info.exchanges[idx]
        values = info.price_volatility(idx, window)
        iterations = range(window, len(values) + window)
        
        plt.plot(iterations, values, color='black', label=exchange.name)

    # plot N exchanges
    else:
        for k, v in info.exchanges.items():
            values = info.price_volatility(k, window)
            iterations = range(window, len(values) + window)

            plt.plot(iterations, values, label=v.name)

    plt.legend()
    plt.show()


def plot_volatility_return(
        info:    SimulatorInfo,
        idx:     int   = None,
        window:  int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot stock return volatility

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id, defaults to None (all exchanges)
    :param window: sample size to calculate std, > 1, defaults to 5
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title(f'Stock Return Volatility (WINDOW: {window})')
    plt.xlabel('Iterations')
    plt.ylabel('Return Volatility')

    # plot 1 exchange
    if idx is not None:
        exchange = info.exchanges[idx]
        values = info.return_volatility(idx, window)
        iterations = range(window, len(values) + window)
        
        plt.plot(iterations, values, color='black', label=exchange.name)

    # plot N exchanges
    else:
        for k, v in info.exchanges.items():
            values = info.return_volatility(k, window)
            iterations = range(window, len(values) + window)

            plt.plot(iterations, values, label=v.name)

    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from AgentBasedModel.utils import math


def plot_two_assets(
        info,
        idx_top: int,          # id актива, который рисуем наверху
        idx_bot: int,          # id актива внизу
        *,                     # дальше только именованные параметры
        rolling: int = 1,
        spread:  bool = False,
        figsize: tuple = (7, 6)
    ):
    """
    Рисует **два** независимых подграфа (один под другим) для указанных
    рынков.  Каждый график имеет собственную ось‑X той же длины, что и
    его временной ряд – поэтому никаких NaN‑ов и ошибок размерностей
    не возникает, даже если warm‑up у активов различается.
    """

    # ── данные верхнего графика ───────────────────────────────────────
    y_top   = math.rolling(info.prices[idx_top],  rolling)
    it_top  = range(rolling-1, rolling-1 + len(y_top))
    ex_top  = info.exchanges[idx_top]

    # ── данные нижнего графика ────────────────────────────────────────
    y_bot   = math.rolling(info.prices[idx_bot],  rolling)
    it_bot  = range(rolling-1, rolling-1 + len(y_bot))
    ex_bot  = info.exchanges[idx_bot]

    # ── фигура из двух подграфов ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=False)

    # ---------- верхний ------------------------------------------------
    ax1.plot(it_top, y_top, color='tab:blue', label=f'{ex_top.name} mid')
    if spread:
        bid = math.rolling([s['bid'] for s in info.spreads[idx_top]], rolling)
        ask = math.rolling([s['ask'] for s in info.spreads[idx_top]], rolling)
        ax1.plot(it_top, bid, ls='--', color='tab:green', alpha=.35, label='bid')
        ax1.plot(it_top, ask, ls='--', color='tab:red',   alpha=.35, label='ask')
    ax1.set_title(f'{ex_top.name}  (MA {rolling})')
    ax1.set_ylabel('Price')
    ax1.legend()

    # ---------- нижний -------------------------------------------------
    ax2.plot(it_bot, y_bot, color='tab:orange', label=f'{ex_bot.name} mid')
    if spread:
        bid = math.rolling([s['bid'] for s in info.spreads[idx_bot]], rolling)
        ask = math.rolling([s['ask'] for s in info.spreads[idx_bot]], rolling)
        ax2.plot(it_bot, bid, ls='--', color='tab:green', alpha=.35, label='bid')
        ax2.plot(it_bot, ask, ls='--', color='tab:red',   alpha=.35, label='ask')
    ax2.set_title(f'{ex_bot.name}  (MA {rolling})')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Price')
    ax2.legend()

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from AgentBasedModel.simulator import SimulatorInfo
import AgentBasedModel.utils.math as math


import matplotlib.pyplot as plt
from AgentBasedModel.simulator import SimulatorInfo
import AgentBasedModel.utils.math as math


def plot_two_assets_stacked(
        info,
        idx_top: int,
        idx_bot: int,
        *,
        rolling: int = 1,
        spread: bool = False,
        figsize: tuple = (7, 6)
    ):
    """
    Рисует **два** независимых подграфа (один под другим) для указанных
    рынков одинаковой длины: короткий ряд смещается вправо на разницу.
    """
    # Данные верхнего графика
    y_top = math.rolling(info.prices[idx_top], rolling)
    len_top = len(y_top)
    ex_top = info.exchanges[idx_top]

    # Данные нижнего графика
    y_bot = math.rolling(info.prices[idx_bot], rolling)
    len_bot = len(y_bot)
    ex_bot = info.exchanges[idx_bot]

    # Определяем общую длину и смещения
    max_len = max(len_top, len_bot)
    shift_top = max_len - len_top
    shift_bot = max_len - len_bot

    # Создаем оси
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Верхний график
    x_top = [shift_top + i for i in range(len_top)]
    ax1.plot(x_top, y_top, label=f'{ex_top.name} mid')
    if spread:
        bids = math.rolling([s['bid'] for s in info.spreads[idx_top]], rolling)
        asks = math.rolling([s['ask'] for s in info.spreads[idx_top]], rolling)
        x_b = [shift_top + i for i in range(len(bids))]
        ax1.plot(x_b, bids, ls='--', label='bid')
        ax1.plot(x_b, asks, ls='--', label='ask')
    ax1.set_title(f'{ex_top.name} (MA {rolling})')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Нижний график
    x_bot = [shift_bot + i for i in range(len_bot)]
    ax2.plot(x_bot, y_bot, label=f'{ex_bot.name} mid')
    if spread:
        bids = math.rolling([s['bid'] for s in info.spreads[idx_bot]], rolling)
        asks = math.rolling([s['ask'] for s in info.spreads[idx_bot]], rolling)
        x_b = [shift_bot + i for i in range(len(bids))]
        ax2.plot(x_b, bids, ls='--', label='bid')
        ax2.plot(x_b, asks, ls='--', label='ask')
    ax2.set_title(f'{ex_bot.name} (MA {rolling})')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Price')
    ax2.legend()

    # Устанавливаем общий диапазон по X
    ax2.set_xlim(0, max_len - 1)
    plt.tight_layout()
    plt.show()




