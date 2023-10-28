import re

import numpy as np

from statistical_simulation_tools import DistributionFitter
from statistical_simulation_tools.utils import get_distributions, sturges_bins

data = np.random.normal(10, 2, size=(10000,))

print(get_distributions())  # Available distributions to fit

desired_distributions = [
    distribution for distribution in get_distributions() if re.search("norm", distribution)
]  # Get the distribution from the normal family


distribution_fitter = DistributionFitter(
    distributions=desired_distributions, bins=sturges_bins(data)
)

distribution_fitter.fit(data)

print(distribution_fitter.summary(sort_by="ks_statistic"))
