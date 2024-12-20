from ..module_loader import ModuleLoader
from bunch import Bunch
import pytorch_common.util as pu
import model as ml


class GMFLoader(ModuleLoader):
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
            'gmf',
            user_seq_col,
            user_seq_col,
            item_seq_col,
            update_period_in_min,
            disable_plot
        )


    def create_model(self, dataset):
        params = Bunch({
            'model': Bunch({
                'n_users'       : dataset[self._user_seq_col].unique().shape[0],
                'n_items'       : dataset[self._item_seq_col].unique().shape[0],
                'device'        : pu.get_device(),
                'embedding_size': 50,
                'weights_path'  : self._weights_path
            }),
            'train': Bunch({
                'lr'         : 0.001,
                'lr_factor'  : 0.03,
                'lr_patience': 5,
                'epochs'     : 7,
                'n_workers'  : 24,
                'batch_size' : 128,
                'eval_percent': 0.1
            }),
            'metrics': Bunch({
                'experiment' : self._predictor_name,
                'path'       : self._metrics_path,
                'n_samples'  : 250,
                'batch_size' : 2000
            })
        })
        return ml.GMF(
            n_users        = params.model.n_users,
            n_items        = params.model.n_items,
            embedding_size = params.model.embedding_size
        ).to(params.model.device), params
