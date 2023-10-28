import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import kstest, rv_continuous

logger = logging.getLogger(__name__)

DistributionParameters = Dict[str, Union[int, float]]


@dataclass(init=True, repr=True, eq=True)
class DistributionFitterResult:
    """A class to store the results of a distribution fitting."""

    distribution: str
    fitted_pdf: np.ndarray
    squared_error: float
    aic: float
    bic: float
    ks_statistic: float
    ks_p_value: float
    fitted_params: DistributionParameters = field(default_factory=dict)


class DistributionFitter:
    def __init__(self, distributions: List[str], bins: int = 100, kde: bool = True) -> None:
        self._data: Optional[np.ndarray] = None
        self._results: Dict[str, DistributionFitterResult] = {}
        self._distributions: List[str] = distributions
        self._bins: int = bins
        self._kde: bool = kde
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Check if the distribution fitter is fitted."""
        return self._is_fitted

    def validate_distribution(self, distribution_name: str) -> bool:
        """Validate that a distribution has been fitted."""
        if not self._is_fitted:
            raise ValueError("DistributionFitter must be fitted before validating distributions.")
        return distribution_name in self._distributions

    @staticmethod
    def _trim_data(
        data: np.ndarray, lower_bound: float = -np.inf, upper_bound: float = np.inf
    ) -> np.ndarray:
        """Trim data based on lower and upper bounds."""
        return data[(data >= lower_bound) & (data <= upper_bound)]

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
            try:
                fitted_params: DistributionParameters = self.fit_single_distribution(
                    data=data_trimmed, distribution_name=distribution
                )
                fitted_pdf: np.ndarray = self.get_pdf(data_trimmed, distribution, fitted_params)
                goodness_of_fit_metrics: DistributionFitterResult = (
                    self.get_goodness_of_fit_metrics(
                        data=data_trimmed,
                        params=fitted_params,
                        fitted_pdf=fitted_pdf,
                        distribution_name=distribution,
                    )
                )

                self._results[distribution] = goodness_of_fit_metrics
            except Exception as e:
                logger.error("Error while fitting distribution: %s", e)
                self._results[distribution] = DistributionFitterResult(
                    distribution=distribution,
                    fitted_pdf=np.zeros(self._bins),
                    squared_error=np.inf,
                    aic=np.inf,
                    bic=np.inf,
                    ks_statistic=np.inf,
                    ks_p_value=np.inf,
                    fitted_params={},
                )
        # Changing the state of the object to fitted
        self._is_fitted = True

    @staticmethod
    def fit_single_distribution(
        data: np.ndarray, distribution_name: str, **kwargs
    ) -> DistributionParameters:
        distribution: rv_continuous = getattr(scipy.stats, distribution_name)
        logger.debug("Fitting distribution: %s", distribution_name)
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

    def get_distribution(self, distribution_name: str) -> rv_continuous:
        assert self._is_fitted, "You need to fit the distribution first"

        return getattr(scipy.stats, distribution_name)(
            **self._results[distribution_name].fitted_params
        )

    def get_distribution_parameters(self, distribution_name: str) -> DistributionParameters:
        assert self._is_fitted, "You need to fit the distribution first"

        return self._results[distribution_name].fitted_params

    def get_goodness_of_fit_metrics(
        self,
        data: np.ndarray,
        params: DistributionParameters,
        fitted_pdf: np.ndarray,
        distribution_name: str,
    ) -> DistributionFitterResult:
        """
        Extract the goodness of fit metrics for a given distribution and its parameters

        """

        distribution: rv_continuous = getattr(scipy.stats, distribution_name)
        y_hist, x_hist = self.get_histogram(data=data, bins=self._bins, density=self._kde)

        logLik = np.sum(distribution.logpdf(x_hist, **params))
        k = len(params)
        n = len(data)

        error_sum_of_squares = np.sum((fitted_pdf - y_hist) ** 2)
        aic = 2 * k - 2 * logLik
        bic = k * np.log(n) - 2 * logLik

        dist_fitted = distribution(**params)
        ks_statistic, ks_p_value = kstest(data, dist_fitted.cdf)

        return DistributionFitterResult(
            distribution=distribution_name,
            fitted_params=params,
            fitted_pdf=fitted_pdf,
            squared_error=error_sum_of_squares,
            aic=aic,
            bic=bic,
            ks_statistic=ks_statistic,
            ks_p_value=ks_p_value,
        )

    def summary(self, sort_by: Optional[str] = None, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Returns a summary of the fitted distributions

        param sort_by: The column to sort by
        param top_n: The number of rows to return

        return: A pandas dataframe with the summary
        """

        assert self._is_fitted, "You need to fit the distribution first"

        sort_by = sort_by if sort_by is not None else "squared_error"

        summary = (
            pd.DataFrame.from_dict(self._results, orient="index")
            .drop(columns=["fitted_pdf"])
            .sort_values(by=sort_by)
        )

        if top_n is not None:
            summary = summary.head(top_n)
        return summary

    def fit_distribution_by_factor(
        self,
        df: pd.DataFrame,
        factor: str,
        variable: str,
        distribution_name: str,
        minimum_number_of_observations: Optional[int] = None,
        **kwargs
    ) -> List[Dict]:
        factor_levels = df[factor].unique().tolist()
        distribution: rv_continuous = getattr(scipy.stats, distribution_name)

        minimum_number_of_observations = (
            minimum_number_of_observations if minimum_number_of_observations is not None else 0
        )

        default_parameters = self.get_distribution_parameters(distribution_name)

        fitted_distributions = []

        for factor_level in factor_levels:
            pre_trimmed_data = df[df[factor] == factor_level][variable].to_numpy()
            upper_bound = kwargs.get("upper_bound", None)
            lower_bound = kwargs.get("lower_bound", None)

            data_trimmed = self._trim_data(pre_trimmed_data, lower_bound, upper_bound)

            number_observations = len(data_trimmed)
            if number_observations > minimum_number_of_observations:
                parameters_names = (
                    (distribution.shapes + ", loc, scale").split(", ")
                    if distribution.shapes
                    else ["loc", "scale"]
                )
                estimated_parameters = distribution.fit(data=data_trimmed)
                response = {
                    "factor_level": factor_level,
                    "distribution": distribution_name,
                    "parameters": {
                        param_k: param_v
                        for param_k, param_v in zip(parameters_names, estimated_parameters)
                    },
                }
            else:
                response = {
                    "factor_level": factor_level,
                    "distribution": distribution_name,
                    "parameters": default_parameters,
                }

            fitted_distributions.append(response)

        return fitted_distributions
