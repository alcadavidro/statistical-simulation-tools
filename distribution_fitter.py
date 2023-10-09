import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import entropy as kl_div
from scipy.stats import kstest, rv_continuous

logger = logging.getLogger(__name__)


DistributionParameters = Dict[str, Union[int, float]]


@dataclass(init=True, repr=True, eq=True)
class DistributionFitterResult:
    distribution: str
    fitted_pdf: np.ndarray
    squared_error: float
    aic: float
    bic: float
    kullberg_divergence: float
    ks_statistic: float
    ks_p_value: float
    fitted_params: Dict[str, Union[int, float]] = field(default_factory=dict)


class DistributionFitter:
    def __init__(
        self,
        distributions: List[str],
        bins: int = 100,
        kde: bool = True,
    ) -> None:
        self._data: np.ndarray = None
        self._results: Dict[str, Any] = {}
        self._distributions = distributions
        self._bins = bins
        self._kde = kde

    @staticmethod
    def _trim_data(
        data: np.ndarray,
        lower_bound: Optional[Union[int, float]] = None,
        upper_bound: Optional[Union[int, float]] = None,
    ) -> np.ndarray:
        upper_bound = upper_bound if upper_bound is not None else data.max()
        lower_bound = lower_bound if lower_bound is not None else data.min()

        data_trimmed = data[np.logical_and(data >= lower_bound, data <= upper_bound)]
        return data_trimmed

    @staticmethod
    def get_histogram(
        data: np.ndarray, bins: int = 100, density: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Extracting the histogram
        y, x = np.histogram(data, bins=bins, density=density)
        # np.histogram returns N + 1 observations as the locations of the bards
        # by doing this we take the coordinates in the middle of the bar
        x = np.array([(value + x[i + 1]) / 2.0 for i, value in enumerate(x[0:-1])])

        # Return the data to be used as histogram for fitting
        return y, x

    def fit(self, data: np.ndarray, **kwargs):
        upper_bound = kwargs.get("upper_bound", None)
        lower_bound = kwargs.get("lower_bound", None)

        data_trimmed = self._trim_data(data, lower_bound, upper_bound)
        # Storing the data used to fit for later use by the analyst
        self._data = data_trimmed

        for distribution in self._distributions:
            fitted_params: DistributionParameters = self.fit_single_distribution(
                data=data_trimmed, distribution_name=distribution
            )
            fitted_pdf: np.ndarray = self.get_pdf(data_trimmed, distribution, fitted_params)
            goodness_of_fit_metrics: DistributionFitterResult = self.get_goodness_of_fit_metrics(
                data=data_trimmed,
                params=fitted_params,
                fitted_pdf=fitted_pdf,
                distribution_name=distribution,
            )

            self._results[distribution] = goodness_of_fit_metrics

    def fit_single_distribution(
        self, data: np.ndarray, distribution_name: str, **kwargs
    ) -> DistributionParameters:
        distribution: rv_continuous = getattr(scipy.stats, distribution_name)
        logger.info("Fitting distribution: %s", distribution_name)
        estimated_parameters = distribution.fit(data=data, **kwargs)

        parameters_names = (
            (distribution.shapes + ", loc, scale").split(", ")
            if distribution.shapes
            else ["loc", "scale"]
        )

        return {
            param_k: param_v for param_k, param_v in zip(parameters_names, estimated_parameters)
        }

    def get_pdf(
        self, data: np.ndarray, distribution_name: str, fitted_params: DistributionParameters
    ) -> np.ndarray:
        distribution: rv_continuous = getattr(scipy.stats, distribution_name)
        _, x = self.get_histogram(data=data, bins=self._bins, density=self._kde)
        pdf = distribution.pdf(x, **fitted_params)
        return pdf

    def get_goodness_of_fit_metrics(
        self,
        data: np.ndarray,
        params: DistributionParameters,
        fitted_pdf: np.ndarray,
        distribution_name: str,
    ) -> DistributionFitterResult:
        distribution: rv_continuous = getattr(scipy.stats, distribution_name)
        y_hist, x_hist = self.get_histogram(data=data, bins=self._bins, density=self._kde)

        logLik = np.sum(distribution.logpdf(x_hist, **params))
        k = len(params)
        n = len(data)

        # Goodness of fit metrics
        ##
        error_sum_of_squares = np.sum((fitted_pdf - y_hist) ** 2)
        aic = 2 * k - 2 * logLik
        bic = k * np.log(n) - 2 * logLik

        # Kullback Leibler divergence
        kullberg_divergence = kl_div(fitted_pdf, y_hist)

        # Calculate kolmogorov-smirnov goodness-of-fit statistic for empirical distribution
        dist_fitted = distribution(**params)
        ks_statistic, ks_p_value = kstest(data, dist_fitted.cdf)

        return DistributionFitterResult(
            distribution=distribution_name,
            fitted_params=params,
            fitted_pdf=fitted_pdf,
            squared_error=error_sum_of_squares,
            aic=aic,
            bic=bic,
            kullberg_divergence=kullberg_divergence,
            ks_statistic=ks_statistic,
            ks_p_value=ks_p_value,
        )

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the fitted distributions
        """
        summary = pd.DataFrame.from_dict(self._results, orient="index")
        return summary
