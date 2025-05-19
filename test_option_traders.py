# main_single_option.py
"""Standalone simulation script: one underlying + one option market.

Параметры‑конфиги легко менять, чтобы пробовать разные составы
трейдеров.  Сначала идёт *warm‑up* только базового актива, затем колл‑
бэк `setup_option` добавляет единственный call‑option рынок и пачку
опционных трейдеров (включая наши новые классы).
"""

from __future__ import annotations

from random import randint
from typing import List, Dict

from AgentBasedModel import Stock, ExchangeAgent, Simulator, OptionExchangeAgent
from AgentBasedModel.exchange import Option
from AgentBasedModel.multitrader import (
    MultiAssetRandomTrader, MultiAssetFundamentalist,
    MultiAssetChartist2D, MultiAssetMarketMaker2D,
)
from AgentBasedModel.optiontraders import (
    RandomOptionTrader, FairValueOptionTrader, SentimentOptionTrader,
    DeltaHedger, FairPriceMarketMaker

)
from AgentBasedModel.visualization import plot_price, plot_two_assets_stacked

# ────────────────────────────────────────────────────────────────────
# 0. Глобальные параметры симуляции
# ────────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 5e-5
BASE_PRICE     = 100
DIVIDEND       = BASE_PRICE * RISK_FREE_RATE

N_WARMUP = 100   # тиков прогрева
N_MAIN   = 200  # тиков основной фазы

# --- Сколько агентов хотим? Легко менять! -----------------------------
TRADER_COUNTS_BASE: Dict[str, int] = {
    "random":           40,
    "fundamentalist":   10,
    "chartist":         10,
    "market_maker":     4,
}

OPTION_TRADER_COUNTS: Dict[str, int] = {
    "random_option":  10,
    "fair_value":     10,
    "sentiment":      1,
    "delta_hedger":   2,
    "fair_mm":        40,
}

# ────────────────────────────────────────────────────────────────────
# 1. Базовый актив, биржа, трейдеры
# ────────────────────────────────────────────────────────────────────
stock      = Stock(DIVIDEND)
stock_mkt  = ExchangeAgent(stock, RISK_FREE_RATE, transaction_cost=0.01)


def build_base_traders(mkt: ExchangeAgent) -> List:
    """Spawn baseline multi‑asset traders for *one* underlying market."""
    res: List = []
    res += [MultiAssetRandomTrader([mkt])
            for _ in range(TRADER_COUNTS_BASE["random"])]
    res += [MultiAssetFundamentalist([mkt], access=1)
            for _ in range(TRADER_COUNTS_BASE["fundamentalist"])]
    res += [MultiAssetChartist2D([mkt])
            for _ in range(TRADER_COUNTS_BASE["chartist"])]
    res += [MultiAssetMarketMaker2D([mkt])
            for _ in range(TRADER_COUNTS_BASE["market_maker"])]
    return res


traders_base = build_base_traders(stock_mkt)

# ────────────────────────────────────────────────────────────────────
# 2. Стартуем симулятор (пока только базовый рынок)
# ────────────────────────────────────────────────────────────────────
sim = Simulator([stock], [stock_mkt], traders_base)

# ────────────────────────────────────────────────────────────────────
# 3. Колл‑бэк: добавляет ОДИН call‑option рынок после прогрева
# ────────────────────────────────────────────────────────────────────
def setup_option(sim: Simulator):
    """Hook executed после warm‑up: создаёт рынок call‑опциона и опционных трейдеров."""

    # --- параметры опциона -------------------------------------------
    S_last = sim.info.exchanges[stock_mkt.id].price()      # последняя цена базового
    T      = N_MAIN                                        # время до экспирации в тиках

    option = Option(
        underlying          = stock,
        strike              = round(100),     # at‑the‑money
        expiry              = T,
        option_type         = "call",
        underlying_exchange = stock_mkt,
    )

    option_mkt = OptionExchangeAgent(
        underlying_exchange = stock_mkt,
        option              = option,
        sim_info            = sim.info,
        transaction_cost    = 0.01,
    )

    # --- опционные трейдеры ------------------------------------------
    tr_option: List = []
    tr_option += [RandomOptionTrader([stock_mkt, option_mkt])
                  for _ in range(OPTION_TRADER_COUNTS["random_option"])]
    tr_option += [FairValueOptionTrader([stock_mkt, option_mkt], sim_info=sim.info)
                  for _ in range(OPTION_TRADER_COUNTS["fair_value"])]
    tr_option += [SentimentOptionTrader([stock_mkt, option_mkt], sim_info=sim.info)
                  for _ in range(OPTION_TRADER_COUNTS["sentiment"])]
    tr_option += [DeltaHedger([stock_mkt, option_mkt], sim_info=sim.info)
                  for _ in range(OPTION_TRADER_COUNTS["delta_hedger"])]
    tr_option += [FairPriceMarketMaker([stock_mkt, option_mkt], sim_info=sim.info)
                  for _ in range(OPTION_TRADER_COUNTS["fair_mm"])]

    # --- подключаем к симулятору -------------------------------------
    sim.add_market(option, option_mkt, tr_option)


# ────────────────────────────────────────────────────────────────────
# 4. Запуск: warm‑up + основная торговля
# ────────────────────────────────────────────────────────────────────
sim.run(
    warmup_steps       = N_WARMUP,
    main_steps         = N_MAIN,
    setup_after_warmup = setup_option,
    silent             = False,
)

info = sim.info
print(stock_mkt.price())
# ────────────────────────────────────────────────────────────────────
# 5. Быстрая визуализация + отчёт по трейдерам
# ────────────────────────────────────────────────────────────────────
plot_price(info, idx=stock_mkt.id, rolling=1)           # базовый актив

print("ID\tType\tRealPnL_before\tAssets_before\tAssets_after\tRealPnL_final")

results = []
for tid, trader in info.traders.items():
    # --- реализованный PnL: серия ведётся в info.realized_pnl[tid]
    pnl_before  = info.realized_pnl[tid][-4] if len(info.realized_pnl[tid]) > 2 else info.realized_pnl[tid][-1]
    pnl_final   = info.realized_pnl[tid][-1]

    ass_before  = sim.info.assets[tid][-4] if len(info.assets[tid]) > 1 else info.assets[tid][-1]
    ass_final   = sim.info.assets[tid][-1]

    results.append((tid, trader.type, pnl_before, ass_before, ass_final, pnl_final))

# сортируем по финальному реализованному PnL
results.sort(key=lambda x: x[5], reverse=True)

for tid, typ, pnl_b, ass_b, ass_f, pnl_f in results:
    print(f"{tid}\t{typ:20}\t{pnl_b:10.2f}\t{ass_b}\t{ass_f}\t{pnl_f:10.2f}")

# --- визуализация опциона
opt_id = next(ex.id for ex in sim.exchanges if ex.asset.type == 'Option')
plot_price(info, idx=opt_id, rolling=1, spread=True)


plot_two_assets_stacked(
    info,
    idx_top = stock_mkt.id,
    idx_bot = opt_id,
    rolling = 1,
    spread  = True,
    figsize = (8, 6)
)

