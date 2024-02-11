from ..module_loader import ModuleLoader
from bunch import Bunch
import pytorch_common.util as pu
import model as ml
import logging


class DeepFMLoader(ModuleLoader):
    def __init__(
        self,
        weights_path,
        metrics_path,
        tmp_path,
        user_seq_col         : str = 'user_seq',
        item_seq_col         : str = 'item_seq',
        rating_col           : str = 'rating',
        update_period_in_min : int = 180,
        disable_plot               = False,
        params              = Bunch({
            'model': Bunch({
                'units_per_layer'   : [50, 10 ,1],
                'dropout'           : 0,
                'embedding_size'    : 64
            }),
            'train': Bunch({
                'lr'         : 0.001,
                'lr_factor'  : 0.1,
                'lr_patience': 8,
                'epochs'     : 4,
                'n_workers'  : 24,
                'batch_size' : 64,
                'eval_percent': 0.15
            }),
            'metrics': Bunch({
                'n_samples'  : 250,
                'batch_size' : 2000
            })
        })
    ):
        super().__init__(
            weights_path,
            metrics_path,
            tmp_path,
            'deep_fm',
            user_seq_col,
            item_seq_col,
            rating_col,
            update_period_in_min,
            disable_plot
        )
        self.params = params


    def create_model(self, dataset):
        if 'model' not in self.params:
            self.params.model = Bunch()

        self.params.model.features_n_values = [
            dataset[self._user_seq_col].max()+1,
            dataset[self._item_seq_col].max()+1
        ]

        self.params.model.device       = pu.get_device()
        self.params.model.weights_path = self._weights_path

        if 'metrics' not in self.params:
            self.params.metrics = Bunch()

        self.params.metrics.experiment = self._predictor_name
        self.params.metrics.path      =  self._metrics_path

        model = ml.DeepFM(
            self.params.model.features_n_values,
            self.params.model.embedding_size,
            self.params.model.units_per_layer,
            self.params.model.dropout
        ).to(self.params.model.device)

        logging.info(model)

        return model, self.params