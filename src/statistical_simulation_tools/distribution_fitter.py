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
    """A class to store the results of a distribution fitting.

    Attributes:
        distribution (str): The name of the distribution.
        fitted_pdf (np.ndarray): The fitted probability density function values.
        squared_error (float): The squared error of the fit.
        aic (float): Akaike Information Criterion value.
        bic (float): Bayesian Information Criterion value.
        ks_statistic (float): Kolmogorov-Smirnov statistic value.
        ks_p_value (float): Kolmogorov-Smirnov p-value.
        fitted (bool): Indicates whether the distribution was successfully fitted. Defaults to False.
        fitted_params (DistributionParameters): The parameters of the fitted distribution. Defaults to an empty dictionary.
    """

    distribution: str
    fitted_pdf: np.ndarray
    squared_error: float
    aic: float
    bic: float
    ks_statistic: float
    ks_p_value: float
    fitted: bool = field(default=False)
    fitted_params: DistributionParameters = field(default_factory=dict)


class DistributionFitter:
    """A class for fitting distributions to data.

    Attributes:
        distributions (List[str]): List of distribution names to be fitted.
        bins (int): Number of bins to use for histograms. Defaults to 100.
        kde (bool): Indicates whether to use kernel density estimation for histograms. Defaults to True.
    """

    def __init__(self, distributions: List[str], bins: int = 100, kde: bool = True) -> None:
        """Initialize the DistributionFitter with a list of distributions, number of bins, and kde option."""
        self._data: Optional[np.ndarray] = None
        self._results: Dict[str, DistributionFitterResult] = {}
        self._distributions: List[str] = distributions
        self._bins: int = bins
        self._kde: bool = kde
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Check if the distribution fitter is fitted.

        Returns:
            bool: True if the distribution fitter is fitted, False otherwise.
        """
        return self._is_fitted

    def validate_distribution(self, distribution_name: str) -> bool:
        """Validate that a distribution has been fitted successfully.

        Args:
            distribution_name (str): Name of the distribution to validate.

        Returns:
            bool: True if the distribution is valid and has been fitted successfully.

        Raises:
            ValueError: If the distribution fitter is not fitted or if the distribution is not valid or has not been fitted successfully.
        """
        if not self._is_fitted:
            raise ValueError("DistributionFitter must be fitted before validating distributions.")

        if distribution_name not in self._results:
            raise ValueError(
                f"Distribution {distribution_name} is not recognized or has not been fitted."
            )

        distribution_results = self._results.get(distribution_name)
        if distribution_results is None or not distribution_results.fitted:
            raise ValueError(
                f"Distribution {distribution_name} got an error in the fitting process and is not available."
            )

        return True

    def get_distribution(self, distribution_name: str) -> rv_continuous:
        """Get a scipy.stats distribution object for a fitted distribution.

        Args:
            distribution_name (str): Name of the distribution to retrieve.

        Returns:
            rv_continuous: A scipy.stats distribution object.

        Raises:
            RuntimeError: If the distribution fitter is not fitted.
            AssertionError: If no fitted parameters are found for the specified distribution.
        """
        if not self._is_fitted:
            raise RuntimeError("You need to fit the distribution first.")
        self.validate_distribution(distribution_name)
        distribution_params = self._results[distribution_name].fitted_params
        assert (
            distribution_params is not None
        ), f"No fitted parameters found for distribution {distribution_name}."
        return getattr(scipy.stats, distribution_name)(**distribution_params)

    def get_distribution_parameters(self, distribution_name: str) -> DistributionParameters:
        """Get the fitted parameters for a distribution.

        Args:
            distribution_name (str): Name of the distribution whose parameters are to be retrieved.

        Returns:
            DistributionParameters: A dictionary of the fitted parameters.

        Raises:
            RuntimeError: If the distribution fitter is not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("You need to fit the distribution first.")

        self.validate_distribution(distribution_name)
        return self._results[distribution_name].fitted_params

    def results(self) -> Dict[str, DistributionFitterResult]:
        """Get the results of the distribution fitting.

        Returns:
            Dict[str, DistributionFitterResult]: A dictionary of the fitting results, keyed by distribution name.

        Raises:
            RuntimeError: If the distribution fitter is not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("You need to fit the distribution first.")
        return self._results

    def best_fit(self, sort_by: str = "ks_statistic") -> Tuple[str, DistributionFitterResult]:
        """Get the best fitting distribution based on a criterion.

        Args:
            sort_by (str): The criterion to sort by. Defaults to "ks_statistic".

        Returns:
            Tuple[str, DistributionFitterResult]: A tuple of the best fitting distribution name and its result.

        Raises:
            RuntimeError: If the distribution fitter is not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("You need to fit the distribution first.")
        best_distribution = min(self._results.items(), key=lambda x: x[1].__dict__[sort_by])
        return best_distribution

    @staticmethod
    def _trim_data(
        data: np.ndarray, lower_bound: float = -np.inf, upper_bound: float = np.inf
    ) -> np.ndarray:
        """Trim the data to remove outliers.

        This function trims the data by removing values below the lower percentile and above the upper percentile.

        Args:
            data (np.ndarray): The data to trim.
            lower_percentile (float): The lower percentile for trimming. Defaults to 1.0.
            upper_percentile (float): The upper percentile for trimming. Defaults to 99.0.

        Returns:
            np.ndarray: The trimmed data.
        """
        return data[(data >= lower_bound) & (data <= upper_bound)]

    @staticmethod
    def get_histogram(
        data: np.ndarray, bins: int = 100, density: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the histogram of the data.

        This function calculates the histogram of the data and can apply kernel density estimation (KDE) if specified.

        Args:
            data (np.ndarray): The data to calculate the histogram for.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The bin edges and the values of the histogram.
        """
        y, x = np.histogram(data, bins=bins, density=density)
        x = (x[:-1] + x[1:]) / 2  # Calculate midpoints of bars
        return y, x

    def fit(
        self,
        data: np.ndarray,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> None:
        """Fit the specified distributions to the data.

        This function fits a list of specified distributions to the data and stores the results.

        Args:
            data (np.ndarray): The data to fit the distributions to.
        """
        lower_bound = data.min() if lower_bound is not None else lower_bound
        upper_bound = data.max() if upper_bound is not None else upper_bound

        data_trimmed = self._trim_data(data, lower_bound, upper_bound)
        self._data = data_trimmed

        for distribution in self._distributions:
            try:
                fitted_params = self.fit_single_distribution(
                    data=data_trimmed, distribution_name=distribution
                )
                fitted_pdf = self.get_pdf(data_trimmed, distribution, fitted_params)
                goodness_of_fit_metrics = self.get_goodness_of_fit_metrics(
                    data=data_trimmed,
                    params=fitted_params,
                    fitted_pdf=fitted_pdf,
                    distribution_name=distribution,
                )
                self._results[distribution] = goodness_of_fit_metrics

            except Exception as e:
                logger.error("Error while fitting distribution %s: %s", distribution, e)
                self._results[distribution] = DistributionFitterResult(
                    distribution=distribution,
                    fitted_pdf=np.zeros(self._bins),
                    squared_error=np.inf,
                    aic=np.inf,
                    bic=np.inf,
                    ks_statistic=np.inf,
                    ks_p_value=np.inf,
                    fitted=False,
                    fitted_params={},
                )
        self._is_fitted = True

    @staticmethod
    def fit_single_distribution(
        data: np.ndarray, distribution_name: str, **kwargs
    ) -> DistributionParameters:
        """Fit a single distribution to the data.

        This function fits a specified distribution to the data and stores the result.

        Args:
            distribution_name (str): The name of the distribution to fit.
        """
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
        """Get the probability density function (PDF) of a fitted distribution.

        This function calculates the PDF of a fitted distribution for the given data points.

        Args:
            distribution_name (str): The name of the distribution to calculate the PDF for.
            x (np.ndarray): The data points to calculate the PDF at.

        Returns:
            np.ndarray: The PDF values of the distribution at the given data points.

        Raises:
            RuntimeError: If the distribution has not been fitted.
            ValueError: If the distribution is not valid or has not been fitted successfully.
        """
        distribution = getattr(scipy.stats, distribution_name)
        _, x = self.get_histogram(data=data, bins=self._bins, density=self._kde)
        return distribution.pdf(x, **fitted_params)

    def get_goodness_of_fit_metrics(
        self,
        data: np.ndarray,
        params: DistributionParameters,
        fitted_pdf: np.ndarray,
        distribution_name: str,
    ) -> DistributionFitterResult:
        """Calculate goodness-of-fit metrics for a fitted distribution.

        This function calculates various goodness-of-fit metrics for a specified distribution.

        Args:
            distribution_name (str): The name of the distribution to calculate the metrics for.

        Returns:
            DistributionFitterResult: A data container with the calculated goodness-of-fit metrics.

        Raises:
            RuntimeError: If the distribution has not been fitted.
            ValueError: If the distribution is not valid or has not been fitted successfully.
        """
        distribution: rv_continuous = getattr(scipy.stats, distribution_name)
        y_hist, x_hist = self.get_histogram(data=data, bins=self._bins, density=self._kde)

        logLik = np.sum(distribution.logpdf(x_hist, **params))
        k = len(params)
        n = len(data)

        error_sum_of_squares = np.sum((fitted_pdf - y_hist) ** 2)
        aic = 2 * k - 2 * logLik
        bic = n * np.log(error_sum_of_squares / n) + k * np.log(n)

        ks_statistic, ks_p_value = kstest(data, distribution.cdf, args=tuple(params.values()))

        return DistributionFitterResult(
            distribution=distribution_name,
            fitted_pdf=fitted_pdf,
            squared_error=error_sum_of_squares,
            aic=aic,
            bic=bic,
            ks_statistic=ks_statistic,
            ks_p_value=ks_p_value,
            fitted=True,
            fitted_params=params,
        )

    def summary(self, sort_by: Optional[str] = None, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get a summary of the fitted distributions.

        Args:
            sort_by (Optional[str]): The column to sort by. Defaults to None.
            top_n (Optional[int]): The number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame: A pandas dataframe with the summary.

        Raises:
            AssertionError: If the distribution fitter is not fitted.
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
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> List[Dict]:
        """
        Fit a specified distribution to a variable within a DataFrame, grouped by levels of a factor.

        For each level of the factor, the specified distribution is fitted to the data of the variable.
        If the number of observations in a group is below the specified minimum, default parameters
        are used instead.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            factor (str): The column name in the DataFrame representing the factor by which to group the data.
            variable (str): The column name in the DataFrame representing the variable to fit the distribution to.
            distribution_name (str): The name of the distribution to fit.
            minimum_number_of_observations (Optional[int]): The minimum number of observations required to fit the distribution.
                If not provided, it defaults to 0.
            lower_bound (Optional[float]): The lower bound for trimming the data. If not provided, no lower bound is applied.
            upper_bound (Optional[float]): The upper bound for trimming the data. If not provided, no upper bound is applied.

        Returns:
            List[Dict]: A list of dictionaries, each containing:
                - "factor_level": The level of the factor.
                - "distribution": The name of the fitted distribution.
                - "parameters": The estimated parameters of the distribution if the number of observations is above the
                minimum required, otherwise the default parameters of the distribution.
        """
        factor_levels = df[factor].unique()

        minimum_number_of_observations = (
            minimum_number_of_observations if minimum_number_of_observations is not None else 0
        )

        default_parameters = self.get_distribution_parameters(distribution_name)

        fitted_distributions = []

        for factor_level in factor_levels:
            data = df[df[factor] == factor_level][variable].to_numpy()
            lower_bound = data.min() if lower_bound is not None else lower_bound
            upper_bound = data.max() if upper_bound is not None else upper_bound

            data_trimmed = self._trim_data(data, lower_bound, upper_bound)

            number_observations = len(data_trimmed)
            if number_observations > minimum_number_of_observations:
                estimated_parameters = self.fit_single_distribution(
                    data=data_trimmed, distribution_name=distribution_name
                )
                response = {
                    "factor_level": factor_level,
                    "distribution": distribution_name,
                    "parameters": estimated_parameters,
                }
            else:
                response = {
                    "factor_level": factor_level,
                    "distribution": distribution_name,
                    "parameters": default_parameters,
                }

            fitted_distributions.append(response)

        return fitted_distributions
