import functools
import numpy as np
from scipy.special import erf
from scipy.optimize import fmin, curve_fit


def sigmoid(x_arr, x0, L, k, b):
    """
    x0 = x value of the sigmoid midpoint
    L = max value of sigmoid
    k = logistic growth rate aka steepness
    b = y-offset
    """
    return L / (1.0 + np.exp(-k * (x_arr - x0))) + b


def naive_fit(x, y, fit_args):
    # parameter bounds
    bounds = (
        [np.min(x), -np.inf, -np.inf, np.min(y)],
        [np.max(x), np.inf, np.inf, np.inf],
    )
    try:
        popt, pcov = curve_fit(
            sigmoid, x, y, method=fit_args.get("method", "dogbox"), bounds=bounds
        )

    except:
        # If can't fit just put out the same data as original
        print("Error fitting the data")
        popt = [0, 1, -1, 0]

    return popt, pcov


"""
Rest modified from:
https://github.com/int-brain-lab/IBL-pipeline/blob/ce6d79fd711f878b7dd85e7c19b7e1f065f117f2/ibl_pipeline/utils/psychofit.py#L30
"""


def validate_params(params, validate_len=3):
    """
    Raises:
        ValueError: pars must be a vector of length 3
        ValueError: each of the three parameters must be scalar
        TypeError: pars must be a list or numpy array
    """
    if isinstance(params, (list, tuple)):
        params = np.array(params)
    elif not isinstance(params, np.ndarray):
        raise TypeError("params must be a list or numpy array")

    if params.shape[0] != validate_len:
        raise ValueError("params must be a vector of length {0}".format(validate_len))

    if (params[0].size != 1) or (params[1].size != 1) or (params[2].size != 1):
        raise ValueError("each of the three parameters must be scalar")

    return 1


def weibull(params, xx):
    """
    Weibull function from 0 to 1, with lapse rate.
    Args:
        pars: Model parameters [alpha, beta, gamma].
            alpha:  threshold parameter (location of function)
            beta:   slope parameter (rate of change)
            gamma:  lapse rate ()
        xx: vector of stim levels (%).
    Returns:
        A vector of length xx
    """
    if validate_params(params):
        alpha = params[0]
        beta = params[1]
        gamma = params[2]

    wbull = (1 - gamma) - (1 - 2 * gamma) * np.exp(-((xx / alpha) ** beta))
    return wbull


def erf_psycho(params, xx):
    """
    erf function from 0 to 1, with lapse rate.
    Args:
        pars: Model parameters [threshold, slope, gamma].
        xx: vector of stim levels (%).
    Returns:
        ff: A vector of length xx
    """
    if validate_params(params):
        bias = params[0]
        slope = params[1]
        gamma = params[2]

    ef = gamma + (1 - 2 * gamma) * (erf((xx - bias) / slope) + 1) / 2
    return ef


def erf_psycho2(params, xx):
    """
    erf function from 0 to 1, with two lapse rates.
    Args:
        pars: Model parameters [bias, slope, gamma1, gamm2].
        xx: vector of stim levels (%)
    Returns:
        ff: A vector of length xx
    """
    if validate_params(params, validate_len=4):
        bias = params[0]
        slope = params[1]
        gamma1 = params[2]
        gamma2 = params[3]

    erf2 = gamma1 + (1 - gamma1 - gamma2) * (erf((xx - bias) / slope) + 1) / 2
    return erf2


def mle_fit(
    data,
    P_model="weibull",
    side="right",
    parstart=None,
    parmin=None,
    parmax=None,
    nfits=5,
):
    """
    Maximumum likelihood fit of psychometric function.
    Args:
        data: 3 x n matrix where first row corrsponds to stim levels (%),
            the second to number of trials for each stim level (int),
            the third to proportion correct (float between 0 and 1)
        P_model: The psychometric function. Possibilities include 'weibull'
            (DEFAULT), 'weibull50', 'erf_psycho' and 'erf_psycho_2gammas'
        parstart: Non-zero starting parameters, used to try to avoid local
            minima.  The parameters are [threshold, slope, gamma], or if
            using the 'erf_psycho_2gammas' model append a second gamma value.
            Recommended to use a value > 1.
            If None, some reasonable defaults are used.
        parmin: Minimum parameter values.  If None, some reasonable defaults
            are used
        parmax: Maximum parameter values.  If None, some reasonable defaults
            are used
        nfits: the number of fits
    Returns:
        pars: The parameters from the best of the fits
        L: The likliehood of the best fit

    Raises:
        TypeError: data must be a list or numpy array
        ValueError: data must be m by 3 matrix
    """
    # Input validation
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError("data must be a list or numpy array")

    if data.shape[0] != 3:
        raise ValueError("data must be m by 3 matrix")

    if parstart is None:
        slope = 3.0 if side == "right" else -3.0
        if P_model == "erf_psycho2":
            parstart = np.array([np.mean(data[0, :]), slope, 0.05, 1.3])
        else:
            parstart = np.array([np.mean(data[0, :]), slope, 0.05])
    if parmin is None:
        if P_model == "erf_psycho2":
            parmin = np.array([np.min(data[0, :]), -10.0, 0.0, 0.0])
        else:
            parmin = np.array([np.min(data[0, :]), -10.0, 0.0])
    if parmax is None:
        if P_model == "erf_psycho2":
            parmax = np.array([np.max(data[0, :]), 10.0, 0.4, 0.4])
        else:
            parmax = np.array([np.max(data[0, :]), 10.0, 0.4])

    # find the good values in pp (conditions that were effectively run)
    ii = np.isfinite(data[2, :])

    likelihoods = np.zeros(
        nfits,
    )
    pars = np.empty((nfits, parstart.size))

    f = functools.partial(
        neg_likelihood, data=data[:, ii], P_model=P_model, parmin=parmin, parmax=parmax
    )
    for ifit in range(nfits):
        pars[ifit, :] = fmin(f, parstart, disp=False)
        parstart = parmin + np.random.rand(parmin.size) * (parmax - parmin)
        likelihoods[ifit] = -neg_likelihood(
            pars[ifit, :], data[:, ii], P_model, parmin, parmax
        )

    # the values to be output
    L = likelihoods.max()
    iBestFit = likelihoods.argmax()
    return pars[iBestFit, :], L


def neg_likelihood(pars, data, P_model="weibull", parmin=None, parmax=None):
    """
    Negative likelihood of a psychometric function.
    Args:
        pars: Model parameters [threshold, slope, gamma], or if
            using the 'erf_psycho_2gammas' model append a second gamma value.
        data: 3 x n matrix where first row corrsponds to stim levels (%),
            the second to number of trials for each stim level (int),
            the third to proportion correct (float between 0 and 1)
        P_model: The psychometric function. Possibilities include 'weibull'
            (DEFAULT), 'weibull50', 'erf_psycho' and 'erf_psycho_2gammas'
        parmin: Minimum bound for parameters.  If None, some reasonable defaults
            are used
        parmax: Maximum bound for parameters.  If None, some reasonable defaults
            are used
    Returns:
        l: The likliehood of the parameters.  The equation is:
            - sum(nn.*(pp.*log10(P_model)+(1-pp).*log10(1-P_model)))
            See the the appendix of Watson, A.B. (1979). Probability
            summation over time. Vision Res 19, 515-522.

    Raises:
        ValueError: invalid model, options are "weibull",
                    "weibull50", "erf_psycho" and "erf_psycho_2gammas"
        TypeError: data must be a list or numpy array
        ValueError data must be m by 3 matrix
    """
    # Validate input
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError("data must be a list or numpy array")

    if parmin is None:
        parmin = np.array([0.005, 0.0, 0.0])
    if parmax is None:
        parmax = np.array([0.5, 10.0, 0.25])

    if data.shape[0] == 3:
        xx = data[0, :]
        nn = data[1, :]
        pp = data[2, :]
    else:
        raise ValueError("data must be m by 3 matrix")

    # here is where you effectively put the constraints.
    if (any(pars < parmin)) or (any(pars > parmax)):
        l = 10000000
        return l

    dispatcher = {
        "weibull": weibull,
        "erf_psycho": erf_psycho,
        "erf_psycho2": erf_psycho2,
    }
    try:
        probs = dispatcher[P_model](pars, xx)
    except KeyError:
        raise ValueError("invalid model, options are {0}".format(dispatcher.keys()))

    assert (max(probs) <= 1) or (
        min(probs) >= 0
    ), "At least one of the probabilities is not between 0 and 1"

    probs[probs == 0] = np.finfo(float).eps
    probs[probs == 1] = 1 - np.finfo(float).eps

    l = -sum(nn * (pp * np.log(probs) + (1 - pp) * np.log(1 - probs)))
    return l
