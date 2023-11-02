from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .distribution_fitter import DistributionFitter


class DistributionValidator:
    """Class for validating distribution fits from the DistributionFitter."""

    def __init__(self, distribution_fitter: DistributionFitter, *args, **kwargs):
        if not distribution_fitter.is_fitted:
            raise ValueError(
                "The DistributionFitter instance must be fitted before instantiating DistributionValidator."
            )
        self._distribution_fitter = distribution_fitter

    def validate_goodness_of_fit(
        self, distribution_name: str, sample_proportion: float = 0.01, **kwargs
    ) -> None:
        """Validate the goodness of fit for a distribution."""
        theoretical_data, sample_data = self._sample_data(
            distribution_name=distribution_name, sample_proportion=sample_proportion
        )
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(21, 10))

        self._qq_plot(
            ax=axs[0, 0], theoretical_data=theoretical_data, sample_data=sample_data, **kwargs
        )
        self._plot_ecdf(
            ax=axs[0, 1], theoretical_data=theoretical_data, sample_data=sample_data, **kwargs
        )
        self._plot_histogram(
            ax=axs[1, 0], theoretical_data=theoretical_data, sample_data=sample_data, **kwargs
        )

        self._pp_plot(
            ax=axs[1, 1],
            theoretical_data=theoretical_data,
            sample_data=sample_data,
            distribution_name=distribution_name,
            **kwargs,
        )

        suptitle = kwargs.get("suptitle", f"Goodness of Fit for: {distribution_name}")
        fig.suptitle(suptitle)
        plt.show()

    def _sample_data(
        self, distribution_name: str, sample_proportion: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample data from the distribution."""
        sample_size = int(np.ceil(self._distribution_fitter._data.size * sample_proportion))
        sample_data = np.random.choice(self._distribution_fitter._data, size=sample_size)
        theoretical_distribution = self._distribution_fitter.get_distribution(distribution_name)
        theoretical_data = theoretical_distribution.ppf(np.linspace(0.001, 0.999, len(sample_data)))
        return np.sort(theoretical_data), np.sort(sample_data)

    def _qq_plot(
        self, ax: plt.Axes, theoretical_data: np.ndarray, sample_data: np.ndarray, **kwargs
    ) -> plt.Axes:
        """Generate a QQ plot."""
        ax.scatter(theoretical_data, sample_data, c="b", marker="o")
        ax.plot(
            [np.min(sample_data), np.max(sample_data)],
            [np.min(sample_data), np.max(sample_data)],
            color="r",
            linestyle="--",
        )
        ax.set_title(kwargs.get("title", "QQ Plot for Goodness of Fit"))
        ax.set_xlabel(kwargs.get("xlabel", "Theoretical Quantiles"))
        ax.set_ylabel(kwargs.get("ylabel", "Sample Quantiles"))
        ax.grid(True)
        return ax

    def _plot_ecdf(
        self, ax: plt.Axes, theoretical_data: np.ndarray, sample_data: np.ndarray, **kwargs
    ) -> plt.Axes:
        """Generate an ECDF plot."""
        sns.ecdfplot(theoretical_data, label="Theoretical Data", ax=ax)
        sns.ecdfplot(sample_data, label="Sample Data", ax=ax)
        ax.set_title(kwargs.get("title", "ECDF Plot for Goodness of Fit"))
        ax.legend()
        return ax

    def _plot_histogram(
        self, ax: plt.Axes, theoretical_data: np.ndarray, sample_data: np.ndarray, **kwargs
    ) -> plt.Axes:
        """Generate a histogram plot."""
        sns.histplot(
            theoretical_data,
            label="Theoretical Data",
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        sns.histplot(sample_data, label="Sample Data", stat="density", common_norm=False, ax=ax)
        ax.set_title(kwargs.get("title", "Histogram Plot for Goodness of Fit"))
        ax.legend()
        return ax

    def _pp_plot(
        self,
        ax: plt.Axes,
        theoretical_data: np.ndarray,
        sample_data: np.ndarray,
        distribution_name: str,
        **kwargs,
    ) -> plt.Axes:
        """Generate a histogram plot."""
        distribution = self._distribution_fitter.get_distribution(distribution_name)
        # P-P Plot
        ax.plot(np.linspace(0, 1, len(sample_data)), np.linspace(0, 1, len(sample_data)), "r--")
        ax.plot(np.linspace(0, 1, len(sample_data)), distribution.cdf(sample_data), "o")
        ax.set_title("P-P Plot")
        ax.set_xlabel("Empirical Cumulative Probability")
        ax.set_ylabel("Theoretical Cumulative Probability")
        ax.set_title(kwargs.get("title", "Histogram Plot for Goodness of Fit"))
        ax.legend()
        return ax
