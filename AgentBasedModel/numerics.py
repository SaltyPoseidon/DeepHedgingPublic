"""
numerics.py – единый «щит» от NaN, ±Inf и деления на ноль.
Импортируйте нужные функции там, где возникают рисковые вычисления.
"""

from __future__ import annotations
import math

# ──────────────────────────────────────────────────────────────────────
#  Глобальные константы
# ──────────────────────────────────────────────────────────────────────
MIN_PRICE  : float = 1e-6    # любое «ценовое» значение не должно быть ниже
EPS        : float = 1e-8    # знаменатели / log / sqrt
EXP_CLAMP  : float = 50.0    # |x| > 50  →  exp(x) ≈ 5.2e21 (бывает)

# ──────────────────────────────────────────────────────────────────────
#  Маленькие хелперы
# ──────────────────────────────────────────────────────────────────────
def clip_price(x: float) -> float:
    """Не даём цене/дивидендy упасть до нуля или ниже."""
    return max(float(x), MIN_PRICE)


def safe_div(num: float, den: float) -> float:
    """Защищённое деление: num / max(|den|, EPS)."""
    return float(num) / max(abs(float(den)), EPS)


def safe_exp(x: float) -> float:
    """Экспонента с ограничением входа, чтобы не взорваться."""
    return math.exp(max(-EXP_CLAMP, min(EXP_CLAMP, float(x))))


def finite(x: float, default: float = 0.0) -> float:
    """Заменяет NaN/Inf на `default`."""
    return default if not math.isfinite(float(x)) else float(x)
