# -----------------------------------------------------------------------------
# Predictors...
# -----------------------------------------------------------------------------
from .knn.knn_user_based_predictor import KNNUserBasedPredictor
from .knn.knn_item_based_predictor import KNNItemBasedPredictor
from .module_predictor             import ModulePredictor
from .sample_cached_predictor      import SampleCachedPredictor
from .cached_predictor             import CachedPredictor

# Predictors Ensemple...
from .ensemple.combine.ensemple_combine_strategy           import EnsempleCombineStrategy
from .ensemple.combine.impl.mean_ensemble_combine_strategy import MeanEnsempleCombineStrategy
from .ensemple.ensemple_predictor                          import EnsemplePredictor
