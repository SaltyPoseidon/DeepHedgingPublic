from collections import deque
from dataclasses import dataclass

@dataclass
class _Lot:
    qty:  int     # >0 long, <0 short
    cost: float   # цена входа (для short — цена продажи)

class CostBasisFIFO:
    """
    FIFO-трекинг лотов с поддержкой long/short и расчётом
    накопленного реализованного PnL.

    buy(q, p)  – покупка  (long ↑  или покрытие short ↓)
    sell(q, p) – продажа  (long ↓  или открытие short ↑)
    """
    def __init__(self) -> None:
        self._lots: deque[_Lot] = deque()
        self.realized_pnl: float = 0.0

    # ───────────────────────────────────────────────────────── helpers ──
    def _append(self, qty: int, price: float):
        if qty:
            self._lots.append(_Lot(qty, price))

    # ─────────────────────────────────────────────────────── публичный API ──
    def buy(self, qty: int, price: float):
        """Покупка (*qty* > 0). Сначала покрываем шорт-лоты слева."""
        remaining = qty
        while remaining and self._lots and self._lots[0].qty < 0:
            lot = self._lots[0]
            trade = min(remaining, -lot.qty)
            self.realized_pnl += trade * (lot.cost - price)  # short-cover
            lot.qty += trade
            remaining -= trade
            if lot.qty == 0:
                self._lots.popleft()
        self._append(remaining, price)   # открываем/увеличиваем long

    def sell(self, qty: int, price: float):
        """Продажа (*qty* > 0). Сначала закрываем long-лоты слева."""
        remaining = qty
        while remaining and self._lots and self._lots[0].qty > 0:
            lot = self._lots[0]
            trade = min(remaining, lot.qty)
            self.realized_pnl += trade * (price - lot.cost)  # long-close
            lot.qty -= trade
            remaining -= trade
            if lot.qty == 0:
                self._lots.popleft()
        if remaining:                       # открываем/увеличиваем short
            self._append(-remaining, price)
