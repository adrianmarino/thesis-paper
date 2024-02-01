import service as srv
from .dataset_helper import *
from .emb_helper import *
import os
from .job import Job
import pytorch_common.util as pu


class CFEmbUpdateJob(Job):

  def __init__(
    self,
    ctx,
    start_year           = 2004,
    split_year           = 2018,
    int_test_size        = 0.1,
    update_period_in_min = 1,
    int_trigger_diff     = 10,
  ):
    self.ctx = ctx
    self.start_year           = start_year
    self.split_year           = split_year
    self.int_test_size        = int_test_size
    self.update_period_in_min = update_period_in_min
    self.int_trigger_diff     = int_trigger_diff
    super().__init__(
      ctx  = ctx,
      name = self.__class__.__name__,
      trigger_fn = None
    )


  def __call__(self):
    pu.set_device_name('gpu')

    interactions_df = get_interactions()

    last_size = self._cfg['interactions_count'] if self._cfg else 0

    if last_size > 0:
      diff = abs(len(interactions_df) - last_size)
      logging.info(f'Interactions difference: {diff}. Previous: {last_size}, Current: {len(interactions_df)}')
      if diff < self.int_trigger_diff:
        logging.info(f'Wait for trigger diff: {self.int_trigger_diff}')
        return [], []
      logging.info(f'{self.int_trigger_diff} interactions threshold ' +
        'is exceeded. Retrain model an update embedding into chroma-db.')

    dev_set, test_set = build_datasets(
      path            = os.environ['DATASET_PATH'],
      interactions_df = interactions_df,
      start_year      = self.start_year,
      split_year      = self.split_year,
      int_test_size   = self.int_test_size
    )

    model_loader = srv.DeepFMLoader(
        weights_path          = os.environ['WEIGHTS_PATH'],
        metrics_path          = os.environ['METRICS_PATH'],
        tmp_path              = self.tmp_path,
        user_seq_col          = 'user_seq',
        item_seq_col          = 'item_seq',
        update_period_in_min  = self.update_period_in_min,
    )

    model, params = model_loader.load(dev_set, test_set)

    [user_embeddings, item_embeddings] = model.embedding.feature_embeddings

    user_embs = dev_set.pipe(to_entity_embs, 'user_seq', 'user_id', user_embeddings)
    item_embs = dev_set.pipe(to_entity_embs, 'item_seq', 'item_id', item_embeddings)


    self.ctx.users_cf_emb_repository.upsert_many(user_embs)
    self.ctx.items_cf_emb_repository.upsert_many(item_embs)

    self._cfg = { 'interactions_count': len(interactions_df) }

    return user_embs, item_embs