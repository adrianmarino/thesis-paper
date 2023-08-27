from ..module_loader import ModuleLoader
from bunch import Bunch
import pytorch_common.util as pu
import model as ml


class GMFLoader(ModuleLoader):
    def __init__(self,weights_path, metrics_path, tmp_path):
        super().__init__(weights_path, metrics_path, tmp_path, 'bias-gmf')


    def _create_model(self, train_set):
        params = Bunch({
            'model': Bunch({
                'n_users'       : train_set[self._user_seq_col].unique().shape[0],
                'n_items'       : train_set[self._item_seq_col].unique().shape[0],
                'device'        : pu.get_device(),
                'embedding_size': 50,
                'weights_path'  : self._weights_path
            }),
            'train': Bunch({
                'lr'         : 0.001,
                'lr_factor'  : 0.05,
                'lr_patience': 3,
                'epochs'     : 12,
                'n_workers'  : 24,
                'batch_size' : 64
            }),
            'metrics': Bunch({
                'experiment' : self._predictor_name,
                'path'       : self._metrics_path,
                'n_samples'  : 500,
                'batch_size' : 3000
            })
        })
        return ml.GMF(
            n_users        = params.model.n_users,
            n_items        = params.model.n_items,
            embedding_size = params.model.embedding_size
        ).to(params.model.device), params
