from ..module_loader import ModuleLoader
from bunch import Bunch
import pytorch_common.util as pu
import model as ml


class NNFMLoader(ModuleLoader):
    def __init__(
        self,weights_path,
        metrics_path,
        tmp_path,
        user_seq_col         : str = 'user_seq',
        item_seq_col         : str = 'item_seq',
        rating_col           : str = 'rating',
        update_period_in_min : int = 180,
        disable_plot         = False
    ):
        super().__init__(
            weights_path,
            metrics_path,
            tmp_path,
            'nn_fm',
            user_seq_col,
            user_seq_col,
            item_seq_col,
            update_period_in_min,
            disable_plot
        )

    def _create_model(self, dev_set):
        params = Bunch({
            'model': Bunch({
                'features_n_values' : [
                    dev_set[self._user_seq_col].unique().shape[0],
                    dev_set[self._item_seq_col].unique().shape[0]
                ],
                'units_per_layer'   : [50, 10, 1],
                'dropout'           : 0.2,
                'device'            : pu.get_device(),
                'embedding_size'    : 50,
                'weights_path'      : self._weights_path
            }),
            'train': Bunch({
                'lr'         : 0.001,
                'lr_factor'  : 0.1,
                'lr_patience': 8,
                'epochs'     : 15,
                'n_workers'  : 24,
                'batch_size' : 64,
                'eval_percent': 0.1
            }),
            'metrics': Bunch({
                'experiment' : self._predictor_name,
                'path'       : self._metrics_path,
                'n_samples'  : 250,
                'batch_size' : 2000
            })
        })

        return ml.NNMF(
            params.model.features_n_values,
            params.model.embedding_size,
            params.model.units_per_layer,
            params.model.dropout
        ).to(params.model.device), params
