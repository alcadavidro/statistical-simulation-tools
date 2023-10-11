from typing import Union

import numpy as np
import pandas as pd
from .distribution_fitter import DistributionFitter
import matplotlib.pyplot as plt
import seaborn as sns

class DistributionValidator:
    def __init__(self, distribution_fitter: DistributionFitter, data: np.ndarray) -> None:
        self.fitter = distribution_fitter
        self.data = data
        
    def validate_goodness_of_fit(self, distribution_name: str, sample_proportion: float = 0.01) -> None:
        sample_size = int(np.ceil(self.fitter._data.size * sample_proportion))
        
        sample_data = np.random.choice(self.fitter._data, size=sample_size)
        theoretical_distribution = self.fitter.get_distribution(distribution_name=distribution_name)
        theoretical_data = theoretical_distribution.ppf(np.linspace(0.001, 0.999, len(sample_data)))
        
        theoretical_data = np.sort(theoretical_data)
        sample_data = np.sort(sample_data)
        
        return theoretical_data, sample_data
        
        
    @staticmethod
    def _qq_plot(ax: plt.Axes, theoretical_data: np.ndarray, sample_data: np.ndarray, **kwargs) -> plt.Axes:
        
        df = pd.DataFrame({
            "theoretical_data": theoretical_data, "sample_data": sample_data
        })
        
        
        sns.scatterplot(data=df, x="theoretical_data", y="sample_data", c='b', marker='o', **kwargs, ax=ax)
        # ax.plot([np.min(sample_data), np.max(sample_data)], [np.min(sample_data), np.max(sample_data)], color='r', linestyle='--')
        # ax.title('QQ Plot for Goodness of Fit')
        # ax.xlabel('Theoretical Quantiles')
        # ax.ylabel('Sample Quantiles')
        # ax.grid(True)
        
        return ax