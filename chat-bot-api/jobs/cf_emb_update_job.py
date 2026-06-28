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

    current_count = len(interactions_df)
    prev_count = self._cfg.get('interactions_count', 0) if self._cfg is not None else 0
    delta = current_count - prev_count

    if self._cfg is not None and (len(chat_bot_interactions) == 0 or prev_count == current_count):
      logging.info(f"""
================================================================================
[STEP 1] 🛑 TRAINING SKIPPED (NO CHANGES DETECTED)
================================================================================
Reason: The total number of interactions has not changed since the last execution.
  • Current Interactions: {current_count}
  • Previous Interactions: {prev_count}
  • Delta (Difference): {delta}

No retraining required. ChromaDB embeddings and MongoDB predictions are up-to-date.
================================================================================
""")
      return

    logging.info(f"""
================================================================================
[STEP 1] 🚀 DEEP-FM EMBEDDINGS UPDATE INITIATED
================================================================================
Reason: Changes detected in user interactions (New ratings submitted or deleted).
  • Current Interactions: {current_count}
  • Previous Interactions: {prev_count}
  • Delta (Difference): {'+' if delta > 0 else ''}{delta}
================================================================================
""")

    train_set, test_set = self.helper.split_dataset(interactions_df)
    
    logging.info(f"""
[STEP 3] 🧠 TRAINING NEURAL NETWORK (DEEP-FM)
Starting PyTorch DeepFM training. Device set to GPU.
Parameters: Units [20, 1], Dropout: 0.25, Embedding Size: 50.
""")

    model_loader = srv.DeepFMLoader(
        weights_path          = os.environ['WEIGHTS_PATH'],
        metrics_path          = os.environ['METRICS_PATH'],
        tmp_path              = os.environ['TMP_PATH'],
        user_seq_col          = 'user_seq',
        item_seq_col          = 'item_seq',
        update_period_in_min  = 10,
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

    # If this fails with a state_dict error, it means the architecture changed (e.g., new users added to embedding).
    # In that case, we can't load the old checkpoint. We must catch it.
    try:
        model, params = model_loader.load(train_set, test_set)
    except Exception as e:
        if 'size mismatch' in str(e) or 'state_dict' in str(e) or 'Error' in str(e):
            logging.warning(f"Failed to load model weights. Ignoring previous checkpoint and training from scratch. Details: {e}")
            save_path = f"{os.environ['WEIGHTS_PATH']}/deep_fm.pt"
            if os.path.exists(save_path):
                os.remove(save_path)
            model, params = model_loader.load(train_set, test_set)
        else:
            raise e

    self.helper.update_embeddings(model, train_set)

    await self.helper.update_database(train_set, model)

    self._cfg = { 'interactions_count': len(interactions_df) }
