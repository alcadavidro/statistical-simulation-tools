import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import iqr, sem, skew


# Distribution utils
def get_distributions():
    distributions = []
    for this in dir(scipy.stats):
        if "fit" in eval("dir(scipy.stats." + this + ")"):
            distributions.append(this)
    return [distribution for distribution in distributions if distribution != "_fit"]


def get_common_distributions():
    distributions = get_distributions()
    reference_distributions = [
        "cauchy",
        "chi2",
        "expon",
        "exponpow",
        "gamma",
        "lognorm",
        "norm",
        "powerlaw",
        "rayleigh",
        "uniform",
    ]
    common_distributions = [
        dist for dist in reference_distributions if dist in distributions
    ]
    return common_distributions


# Helper methods for estimating the number of bins required
def sturges_bins(data: np.ndarray) -> int:
    n = len(data)
    bins = int(np.ceil(np.sqrt(n)))
    return bins


def scotts_bins(data: np.ndarray) -> int:
    n = len(data)
    std_dev = np.std(data)
    bins = int(np.ceil((3.5 * std_dev) / (n ** (1 / 3))))
    return bins


def freedman_diaconis_bins(data: np.ndarray) -> int:
    n = len(data)
    iqr_value = iqr(data)
    bins = int(np.ceil((2 * iqr_value) / (n ** (1 / 3))))
    return bins


def rice_bins(data: np.ndarray) -> int:
    n = len(data)
    bins = int(np.ceil(2 * np.cbrt(n)))
    return bins


def doanes_bins(data: np.ndarray) -> int:
    n = len(data)
    skewness = skew(data)
    std_err_skew = sem(data) / np.sqrt(n)
    bins = int(1 + np.log2(n) + np.log2(1 + abs(skewness) / std_err_skew))
    return bins


def filterByLast(df: pd.DataFrame, partition_by: str, order_by: str) -> pd.DataFrame:
    return df.assign(
        row_number=df.groupby(partition_by)[order_by].rank(
            method="first", ascending=False
        )
    ).query("row_number == 1")
