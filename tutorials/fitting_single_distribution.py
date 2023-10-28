import seaborn as sns

from statistical_simulation_tools import DistributionFitter
from statistical_simulation_tools.utils import get_distributions

df = sns.load_dataset("diamonds")

data = df["depth"]

print(get_distributions())  # Available distributions to fit

desired_distribution = "norm"  # The name have to be available in scipy

parameters_estimated = DistributionFitter.fit_single_distribution(
    data=data, distribution_name="norm"
)

print(parameters_estimated)
