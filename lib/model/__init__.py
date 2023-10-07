# -----------------------------------------------------------------------------
# Model/Predictor building blocks...
# -----------------------------------------------------------------------------
from .distance.cosine_distance               import CosineDistance
from .distance.distances_matrix              import rows_distance_matrix, plot_rows_distance_matrix
from .neighbors.nearest_neighbors            import NearestNeighbors, NearestNeighborsResult
from .module.batch_dot                       import BatchDot
from .module.categorical_features_lineal     import CategoricalFeaturesLineal
from .module.embedding_factorization_machine import EmbeddingFactorizationMachine
from .module.multi_feature_embedding         import MultiFeatureEmbedding
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Predictors...
# -----------------------------------------------------------------------------
from .predictor.abstract_predictor            import AbstractPredictor
from .predictor.knn.knn_user_based_predictor  import KNNUserBasedPredictor
from .predictor.knn.knn_item_based_predictor  import KNNItemBasedPredictor
from .predictor.knn.knn_predictor_factory     import KNNPredictorFactory, KNNType
from .predictor.module_predictor              import ModulePredictor
from .predictor.sample_cached_predictor       import SampleCachedPredictor
from .predictor.cached_predictor              import CachedPredictor


# Predictors Ensemple...
from .predictor.ensemple.combine.ensemple_combine_strategy           import EnsempleCombineStrategy
from .predictor.ensemple.combine.impl.mean_ensemble_combine_strategy import MeanEnsempleCombineStrategy
from .predictor.ensemple.ensemple_predictor                          import EnsemplePredictor

# Predictor validation utils...
from .module.mse_loss_fn                     import MSELossFn
from .validate.validator                     import Validator, ValidatorSummary
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Models...
# -----------------------------------------------------------------------------
# Collavorative Filtering based way...
from .module.deep_fm                    import DeepFM
from .module.nnmf                       import NNMF
from .module.gmf                        import GMF
from .module.biased_gmf                 import BiasedGMF
from .module.collaborative_auto_encoder import *
#
# Content Based way...
from .module.autoencoder                import *
#
from .module.training.dataset_factory   import DatasetFactory
from .module.training.module_trainer    import ModuleTrainer
# -----------------------------------------------------------------------------


from .clustering.k_medoids  import KMedoisClustering
from .module.mlp            import MultiLayerPerceptron



# -----------------------------------------------------------------------------
# Hyperparameters optimization...
# -----------------------------------------------------------------------------
from .optimization.optuna_trainer import HyperParamsSampler, OptunaTrainer
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# nsemble...
# -----------------------------------------------------------------------------
from .ensemble.fwls import *
# -----------------------------------------------------------------------------