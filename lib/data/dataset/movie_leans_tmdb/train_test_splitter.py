import pandas as pd
import logging
from bunch import Bunch


class TrainTestSplitter:
  def __init__(
    self,
    split_year = 2018,
    cols       = Bunch(
      user_seq    = 'user_seq',
      item_seq    = 'movie_seq',
      rating      = 'user_movie_rating',
      rating_year = 'user_movie_rating_year',
      rating_mean = 'user_movie_rating_mean',
      rating_norm = 'user_movie_rating_norm' 
    )
  ):
    self.split_year = split_year
    self.cols       = cols


  def __call__(
      self,
      dataset,  
      rating_mean_df = pd.DataFrame(), 
      rating_std     = None
  ):
    # ----------
    # Train set:
    # ----------
    df_train = self.query_train_set(dataset)

    # Add mean rating column by user...
    if rating_mean_df.empty:
        rating_mean_df = self.rating_mean_by_user_seq(df_train)

    # Compute users rating std deviation from train_set...
    if rating_std == None:
        rating_std = self.rating_std(df_train)

    df_train = self.join_by_user_seq(df_train, rating_mean_df)

    df_train = self.append_rating_norm(df_train, rating_std)

    # ---------------
    # Validation set:
    # ---------------
    # - Include only movies an used that exists in train set.
    df_eval = self.query_test_set(dataset, df_train, self.split_year)

    df_eval = self.join_by_user_seq(df_eval, rating_mean_df)

    df_eval = self.append_rating_norm(df_eval, rating_std)

    logging.info(f'Train: {(len(df_train)/len(dataset))*100:.2f} % - Test: {(len(df_eval)/len(dataset))*100:.2f} %')

    return df_train, df_eval, rating_mean_df, rating_std



  def query_train_set(self, dataset):
    df_train = dataset[dataset[self.cols.rating_year] < self.split_year]
    return df_train.loc[:, ~df_train.columns.isin([self.cols.rating_mean, self.cols.rating_norm])]



  def query_test_set(self, dataset, df_train, split_year):
    """ 
    Include only movies an used that exists in train set.
    """
    df_eval = dataset[
        (dataset[self.cols.rating_year] >= self.split_year) &
        (dataset[self.cols.user_seq].isin(df_train[self.cols.user_seq].values)) &
        (dataset[self.cols.item_seq].isin(df_train[self.cols.item_seq].values))
    ]
    return df_eval.loc[:, ~df_eval.columns.isin([self.cols.rating_mean, self.cols.rating_norm])]



  def rating_mean_by_user_seq(self, df):
    return df.groupby(self.cols.user_seq)[self.cols.rating] \
            .mean() \
            .reset_index(name=self.cols.rating_mean)


  def rating_std(self, df):
    return df[self.cols.rating].std()


  def join_by_user_seq(self, df_a, df_b):
    return pd.merge(
        df_a,
        df_b,
        how      = 'inner',
        left_on  = [self.cols.user_seq],
        right_on = [self.cols.user_seq]
    ).dropna()

  
  def append_rating_norm(self, df, std):
    """
    Create normalized rattng column...
    """
    df[self.cols.rating_norm] = df.apply(lambda row: round((row[self.cols.rating] - row[self.cols.rating_mean]) / std, 2), axis=1)
    return df