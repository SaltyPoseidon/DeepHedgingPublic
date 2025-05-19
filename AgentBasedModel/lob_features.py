import numpy as np

def aggregate_levels(order_list, depth):
    """
    Суммирует все ордера на одном ценовом уровне, пока не соберёт `depth`.
    Возвращает list[(price, qty)] длиной ≤ depth (asc/desc уже задан порядком).
    """
    levels = []
    cur = order_list.first
    while cur and len(levels) < depth:
        price = cur.price
        qty   = 0
        # группируем по цене
        while cur and cur.price == price:
            qty += cur.qty
            cur = cur.right
        levels.append((price, qty))
    return levels


def build_lob_features(order_book, *, K=40, tick=0.1):
    """
    →  тензор [2, K, 2]  (канал bid, канал ask)
        dim-2: 0 – относительная цена, 1 – log-depth.

    K – сколько ценовых уровней с каждой стороны.
    tick – минимальный шаг цены (нужен для padding).
    """
    mid = (order_book['bid'].first.price + order_book['ask'].first.price) * 0.5

    bid_lvls = aggregate_levels(order_book['bid'], K)
    ask_lvls = aggregate_levels(order_book['ask'], K)

    tensor = np.zeros((2, K, 2), dtype=np.float32)

    # ---------- BID (канал 0) ----------
    prev_p = order_book['bid'].first.price
    for i in range(K):
        if i < len(bid_lvls):
            price, qty = bid_lvls[i]
            prev_p = price
        else:                                   # padding ниже лучшего bid
            price, qty = prev_p - tick, 0.0
            prev_p = price
        tensor[0, i, 0] = (price / mid) - 1.0          # rel-price
        tensor[0, i, 1] = np.log1p(qty)                # log-depth

    # ---------- ASK (канал 1) ----------
    prev_p = order_book['ask'].first.price
    for i in range(K):
        if i < len(ask_lvls):
            price, qty = ask_lvls[i]
            prev_p = price
        else:                                   # padding выше лучшего ask
            price, qty = prev_p + tick, 0.0
            prev_p = price
        tensor[1, i, 0] = (price / mid) - 1.0
        tensor[1, i, 1] = np.log1p(qty)

    return tensor        # shape (2, K, 2)
