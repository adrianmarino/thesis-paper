import service as srv
import os
from .job import Job
import pytorch_common.util as pu
import pandas as pd
import torch
import data.dataset as ds
import data as dt
import logging
import util as ut
from models import EntityEmb
from bunch import Bunch
import logging
import sys
from .cf_emb_update_helper import CFEmbUpdateJobHelper
import logging


class CFEmbUpdateJob(Job):

  def __init__(
    self,
    ctx,
    update_period_in_min = 1,
  ):
    self.ctx = ctx
    self.update_period_in_min = update_period_in_min
    super().__init__(
      ctx        = ctx,
      name       = self.__class__.__name__,
      trigger_fn = None
    )
    self.helper = CFEmbUpdateJobHelper(ctx)


  async def __call__(self):
    pu.set_device_name('gpu')

    interactions_df = self.helper.get_interactions()

    chat_bot_interactions = interactions_df[interactions_df["user_id"].str.contains("@")]

    if len(chat_bot_interactions) == 0 or self._cfg.get('interactions_count', 0) == len(interactions_df):
      logging.warn("""
        No changes were found in user interactions, therefore updating
        user and item embeddings, as well as predictions for ratings
        of items not seen by users, is not necessary.
      """)
      return

    train_set, test_set = self.helper.split_dataset(interactions_df)

    model_loader = srv.DeepFMLoader(
        weights_path          = os.environ['WEIGHTS_PATH'],
        metrics_path          = os.environ['METRICS_PATH'],
        tmp_path              = os.environ['TMP_PATH'],
        user_seq_col          = 'user_seq',
        item_seq_col          = 'item_seq',
        update_period_in_min  = 1,
        params              = Bunch({
            'model': Bunch({
                'units_per_layer'   : [20, 1],
                'dropout'           : 0.25,
                'embedding_size'    : 50,
            }),
            'train': Bunch({
                'lr'         : 0.001,
                'lr_factor'  : 0.1,
                'lr_patience': 6,
                'epochs'     : 7,
                'n_workers'  : 24,
                'batch_size' : 2048,
                'eval_percent': 0.15
            }),
            'metrics': Bunch({
                'n_samples'  : 250,
                'batch_size' : 2000
            })
        })
    )

    model, params = model_loader.load(train_set, test_set)

    self.helper.update_embeddings(model, train_set)

    await self.helper.update_database(train_set, model)

    self._cfg = { 'interactions_count': len(interactions_df) }
