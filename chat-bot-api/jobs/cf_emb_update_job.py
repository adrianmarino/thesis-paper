import service as srv
import os
from .job import Job
import pytorch_common.util as pu
import rest
import pandas as pd
import torch
import data.dataset as ds
import data as dt
import logging
import util as ut
from models import EntityEmb
from bunch import Bunch 


class CFEmbUpdateJob(Job):

  def __init__(
    self,
    ctx,
    split_year           = 2018,
    update_period_in_min = 1,
    int_trigger_diff     = 10,
  ):
    self.ctx = ctx
    self.split_year           = split_year
    self.update_period_in_min = update_period_in_min
    self.int_trigger_diff     = int_trigger_diff
    self.api_client           = rest.RecChatBotV1ApiClient()
    super().__init__(
      ctx        = ctx,
      name       = self.__class__.__name__,
      trigger_fn = None
    )


  def __call__(self):
    pu.set_device_name('gpu')

    interactions_df = pd.DataFrame(self.api_client.interactions())
    if not self.__must_update(interactions_df):
      return [], []

    # Create year columns...
    interactions_df['year'] = interactions_df['timestamp'].apply(pd.to_datetime).dt.year

    # Generate sequences....
    user_seq = dt.Sequencer('user_id', 'user_seq')
    item_seq = dt.Sequencer('item_id', 'item_seq')

    interactions_df = user_seq.perform(interactions_df)
    interactions_df = item_seq.perform(interactions_df)

    data_splitter = ds.TrainTestSplitter(
      split_year = self.split_year,
      cols       = Bunch(
        user_seq    = 'user_seq',
        item_seq    = 'item_seq',
        rating      = 'rating',
        rating_year = 'year',
        rating_mean = 'rating_mean',
        rating_norm = 'rating_norm' 
      )
    )

    train_set, test_set, _, _ = data_splitter(interactions_df)


    model_loader = srv.DeepFMLoader(
        weights_path          = os.environ['WEIGHTS_PATH'],
        metrics_path          = os.environ['METRICS_PATH'],
        tmp_path              = self.tmp_path,
        user_seq_col          = 'user_seq',
        item_seq_col          = 'item_seq',
        update_period_in_min  = self.update_period_in_min,
    )

    model, params = model_loader.load(train_set, test_set)

    user_embs, item_embs = self.__upload(interactions_df, model.embedding.feature_embeddings)

    self._cfg = { 'interactions_count': len(interactions_df) }

    return user_embs, item_embs


  def __must_update(self, interactions_df):
    last_size = self._cfg['interactions_count'] if self._cfg else 0

    if last_size > 0:
      diff = abs(len(interactions_df) - last_size)
      logging.info(f'Interactions difference: {diff}. Previous: {last_size}, Current: {len(interactions_df)}')
      if diff < self.int_trigger_diff:
        logging.info(f'Wait for trigger diff: {self.int_trigger_diff}')
        return False
      logging.info(f'{self.int_trigger_diff} interactions threshold ' +
        'is exceeded. Retrain model an update embedding into chroma-db.')
    else:
      logging.info(f'Train model an update embedding into chroma-db.')

    return True


  def __upload(self, interactions_df, feature_embeddings):
    [user_embeddings, item_embeddings] = feature_embeddings

    def to_entity_embs(df, seq_col, id_col, embeddings):
      seq_to_id = ut.to_dict(df, seq_col, id_col)
      return [
          EntityEmb(
              id  = id,
              emb = embeddings[seq].tolist()
          )
          for seq, id in seq_to_id.items()
      ]

    user_embs = interactions_df.pipe(to_entity_embs, 'user_seq', 'user_id', user_embeddings)
    self.ctx.users_cf_emb_repository.upsert_many(user_embs)
 
    item_embs = interactions_df.pipe(to_entity_embs, 'item_seq', 'item_id', item_embeddings)
    self.ctx.items_cf_emb_repository.upsert_many(item_embs)


    return user_embs, item_embs