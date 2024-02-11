import pandas as pd
import logging
from bunch import Bunch
import data as dt


class TrainTestSplitter:
  def __init__(
    self,
    n_min_interactions = 20,
    test_size          = 0.3,
    cols               = Bunch(
      order_col   = 'timestamp',
      user_id     = 'user_id',
      item_id     = 'item_id',
      rating      = 'rating',
      rating_mean = 'rating_mean',
      rating_norm = 'rating_norm' 
    )
  ):
    self.n_min_interactions = n_min_interactions
    self.test_size          = test_size
    self.cols               = cols



  def __call__(
      self,
      dataset,  
      rating_mean_df = pd.DataFrame(), 
      rating_std     = None
  ):
    train_df, test_df = dt.interactions_train_test_split(
        dataset,
        order_col          = self.cols.order_col,
        user_id_col        = self.cols.user_id,
        item_id_col        = self.cols.item_id,
        n_min_interactions = self.n_min_interactions,
        test_size          = self.test_size
    )

    # Add mean rating column by user...
    if rating_mean_df.empty:
        rating_mean_df = self.rating_mean_by_user_id(train_df)

    # Compute users rating std deviation from train_set...
    if rating_std == None:
        rating_std = self.rating_std(train_df)

    train_df = self.join_by_user_id(train_df, rating_mean_df)

    train_df = self.append_rating_norm(train_df, rating_std)

    # ---------------
    # Validation set:
    # ---------------
    test_df = self.join_by_user_id(test_df, rating_mean_df)

    test_df = self.append_rating_norm(test_df, rating_std)

    logging.info(f'Train: {(len(train_df)/len(dataset))*100:.2f} % - Test: {(len(test_df)/len(dataset))*100:.2f} %')

    return train_df, test_df, rating_mean_df, rating_std


  def rating_mean_by_user_id(self, df):
    return df.groupby(self.cols.user_id)[self.cols.rating] \
            .mean() \
            .reset_index(name=self.cols.rating_mean)


  def rating_std(self, df):
    return df[self.cols.rating].std()


  def join_by_user_id(self, df_a, df_b):
    return pd.merge(
        df_a,
        df_b,
        how      = 'inner',
        left_on  = [self.cols.user_id],
        right_on = [self.cols.user_id]
    )

  
  def append_rating_norm(self, df, std):
    return dt.normalize(
      df,
      self.cols.rating,
      self.cols.rating_norm,
      self.cols.rating_mean,
      std,
    )

