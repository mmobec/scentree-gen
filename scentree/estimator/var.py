from scentree.estimator.stats_base import StatsEstimator
from statsmodels.tsa.vector_ar.var_model import VAR

class VarEstimator(StatsEstimator):
    estimator_class = VAR
