from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .distribution_fitter import DistributionFitter


class DistributionValidator:
    def validate_goodness_of_fit(
        cls,
        distribution_fitter: DistributionFitter,
        distribution_name: str,
        sample_proportion: float = 0.01,
        **kwargs
    ) -> None:
        theoretical_data, sample_data = cls.sample_data(
            distribution_fitter=distribution_fitter,
            distribution_name=distribution_name,
            sample_proportion=sample_proportion,
        )
        fig, ax = plt.subplots(ncols=3, figsize=(21, 6))

        ax[0] = cls.qq_plot(
            ax=ax[0], theoretical_data=theoretical_data, sample_data=sample_data, kwargs=kwargs
        )
        ax[1] = cls.plot_ecdf(
            ax=ax[1], theoretical_data=theoretical_data, sample_data=sample_data, kwargs=kwargs
        )
        ax[2] = cls.plot_histogram(
            ax=ax[2], theoretical_data=theoretical_data, sample_data=sample_data, kwargs=kwargs
        )

        suptitle = kwargs.get('suptitle', 'Goodness of Fit')

        fig.suptitle(suptitle)

        plt.show()

    @classmethod
    def sample_data(
        cls,
        distribution_fitter: DistributionFitter,
        distribution_name: str,
        sample_proportion: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sample_size = int(np.ceil(distribution_fitter._data.size * sample_proportion))

        sample_data = np.random.choice(distribution_fitter._data, size=sample_size)
        theoretical_distribution = distribution_fitter.get_distribution(
            distribution_name=distribution_name
        )
        theoretical_data = theoretical_distribution.ppf(np.linspace(0.001, 0.999, len(sample_data)))

        theoretical_data = np.sort(theoretical_data)
        sample_data = np.sort(sample_data)

        return theoretical_data, sample_data

    @classmethod
    def qq_plot(
        cls, ax: plt.Axes, theoretical_data: np.ndarray, sample_data: np.ndarray, **kwargs
    ) -> plt.Axes:
        title = kwargs.get('title', 'QQ Plot for Goodness of Fit')
        xlabel = kwargs.get('xlabel', 'Theoretical Quantiles')
        ylabel = kwargs.get('ylabel', 'Sample Quantiles')

        ax.scatter(theoretical_data, sample_data, c='b', marker='o')
        ax.plot(
            [np.min(sample_data), np.max(sample_data)],
            [np.min(sample_data), np.max(sample_data)],
            color='r',
            linestyle='--',
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return ax

    @classmethod
    def plot_ecdf(
        cls, ax: plt.Axes, theoretical_data: np.ndarray, sample_data: np.ndarray, **kwargs
    ) -> plt.Axes:
        title = kwargs.get('title', 'ECDF Plot for Goodness of Fit')

        sns.ecdfplot(theoretical_data, label='Theoretical Data', ax=ax)
        sns.ecdfplot(sample_data, label='Sample Data', ax=ax)
        ax.set_title(title)
        ax.legend()
        return ax

    @classmethod
    def plot_histogram(
        cls, ax: plt.Axes, theoretical_data: np.ndarray, sample_data: np.ndarray, **kwargs
    ) -> plt.Axes:
        title = kwargs.get('title', 'Histogram Plot for Goodness of Fit')

        sns.histplot(
            theoretical_data,
            label='Theoretical Data',
            element='step',
            stat='density',
            common_norm=False,
            ax=ax,
        )
        sns.histplot(sample_data, label='Sample Data', stat='density', common_norm=False, ax=ax)
        ax.set_title(title)
        ax.legend()
        return ax
