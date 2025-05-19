import AgentBasedModel.utils.math as math

from scipy.stats import kendalltau
import statsmodels.api as sm


def test_trend_kendall(values, category: bool = False, conf: float = .95) -> bool | dict:
    """
    Kendall’s Tau test.
    H0: No trend exists
    Ha: Some trend exists
    :return: True - trend exist, False - no trend
    """
    iterations = range(len(values))
    tau, p_value = kendalltau(iterations, values)
    if category:
        return p_value < (1 - conf)
    return {'tau': round(tau, 4), 'p-value': round(p_value, 4)}


def test_trend_ols(values) -> dict:
    """
    Linear regression on time.
    H0: No trend exists
    Ha: Some trend exists
    :return: True - trend exist, False - no trend
    """
    x = range(len(values))
    estimate = sm.OLS(values, sm.add_constant(x)).fit()
    return {
        'value': round(estimate.params[1], 4),
        't-stat': round(estimate.tvalues[1], 4),
        'p-value': round(estimate.pvalues[1], 4)
    }


def trend(
        info,
        idx:    int,
        size:   int   = None,
        window: int   = 5,
        conf:   float = .95,
        th:     float = .01
    ) -> bool | list:
    """Test if exist price trend

    :param info: SimulatorInfo
    :param idx: ExchangeAgent id
    :param size: section size to measure state upon, defaults to None
    :param window: (compatibility) rolling window to measure volatility, defaults to 5
    :param conf: OLS p-value threshold, defaults to .95
    :param th: OLS coeff threshold, defaults to .01
    :return: if size is None -> bool, if size is not None -> list
    """
    prices = info.prices[idx][window:]

    if size is None:
        test = test_trend_ols(prices)
        return test['p-value'] < (1 - conf) and abs(test['value']) > th

    res = list()
    for i in range(len(prices) // size):
        test = test_trend_ols(prices[i*size:(i+1)*size])
        res.append(test['p-value'] < (1 - conf) and abs(test['value']) > th)

    return res


def panic(
        info,
        idx:    int,
        size:   int   = None,
        window: int   = 5,
        th:     float = .01
    ) -> bool | list:
    """Test if volatility is high at any iteration

    :param info: SimulatorInfo
    :param idx: ExchangeAgent id
    :param size: section size to measure state upon, defaults to None
    :param window: rolling window to measure volatility, defaults to 5
    :param th: OLS coeff threshold, defaults to .01
    :return: if size is None -> bool, if size is not None -> list
    """
    volatility = info.price_volatility(idx, window)

    if size is None:
        return any(v > th for v in volatility)

    res = list()
    for i in range(len(volatility) // size):
        sl = volatility[i*size:(i+1)*size]
        res.append(any(v > math.mean(volatility) * th for v in sl))
    
    return res


def disaster(
        info,
        idx:    int,
        size:   int   = None,
        window: int   = 5,
        conf:   float = .95,
        th:     float = .01
    ) -> bool | list:
    """Test volatility increase

    :param info: SimulatorInfo
    :param idx: ExchangeAgent id
    :param size: section size to measure state upon, defaults to None
    :param window: rolling window to measure volatility, defaults to 5
    :param conf: OLS p-value threshold, defaults to .95
    :param th: OLS coeff threshold, defaults to .01
    :return: if size is None -> bool, if size is not None -> list
    """
    volatility = info.price_volatility(idx, window)

    if size is None:
        test = test_trend_ols(volatility)
        return test['value'] > th and test['p-value'] < (1 - conf)

    res = list()
    for i in range(len(volatility) // size):
        test = test_trend_ols(volatility[i*size:(i+1)*size])
        res.append(test['value'] > th and test['p-value'] < (1 - conf))

    return res


def mean_rev(
        info,
        idx:    int,
        size:   int   = None,
        window: int   = 5,
        conf:   float = .95,
        th:     float = .01
    ) -> bool | list:
    """Test volatility decrease

    :param info: SimulatorInfo
    :param idx: ExchangeAgent id
    :param size: section size to measure state upon, defaults to None
    :param window: rolling window to measure volatility, defaults to 5
    :param conf: OLS p-value threshold, defaults to .95
    :param th: OLS coeff threshold, defaults to .01
    :return: if size is None -> bool, if size is not None -> list
    """
    volatility = info.price_volatility(idx, window)

    if size is None:
        test = test_trend_ols(volatility)
        return test['value'] < th and test['p-value'] < (1 - conf)

    res = list()
    for i in range(len(volatility) // size):
        test = test_trend_ols(volatility[i*size:(i+1)*size])
        res.append(test['value'] < th and test['p-value'] < (1 - conf))

    return res


def general_states(
        info,
        idx:    int,
        size:   int = 10,
        window: int = 5
    ) -> str | list:
    """Classify market states on the simulation timeline sections

    1) Measure price trend, volatility value, volatility trend with rolling **window**
    2) Divide simulation timeline with non-intersecting section of lenght **size**
    3) Determing market state for each section

    :param info: SimulatorInfo
    :param idx: ExchangeAgent id
    :param size: section size to measure state upon, defaults to 10
    :param window: rolling window to measure volatility, defaults to 5
    :return: One of the following states (with priority as ordered) 'mean-rev' - volatility is decreasing,
    'disaster' - volatility is increasing, 'panic' - volatility is high, 'trend' - exist price trend,
    'stable' - otherwise
    """
    states_trend = trend(info, idx, size)
    states_panic = panic(info, idx, size, window)
    states_disaster = disaster(info, idx, size, window)
    states_mean_rev = mean_rev(info, idx, size, window)

    res = list()
    for t, p, d, mr in zip(states_trend, states_panic, states_disaster, states_mean_rev):
        if mr:
            res.append('mean-rev')
        elif d:
            res.append('disaster')
        elif p:
            res.append('panic')
        elif t:
            res.append('trend')
        else:
            res.append('stable')
    return res
