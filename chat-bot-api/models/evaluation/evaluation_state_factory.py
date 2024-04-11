import util as ut
import numpy as np
import logging
import client
from faker import Faker
import pandas as pd
from .evaluation_state import EvaluationState
import os


class EvaluationStateFactory:
  @classmethod
  def create(
      cls,
      api_client,
      interactions_test_set,
      items,
      hyper_params,
      path,
      recomendation_size = 5,
      max_patience       = {
          40  : 2,
          100 : 3,
          200 : 4,
          300 : 5
      },
      plot_interval      = 10,
      verbose            = False
  ):
    if not verbose: api_client.verbose_off
    profiles, user_ids  = cls._reset_env(api_client, interactions_test_set, items)
    if not verbose: api_client.verbose_on

    return EvaluationState(
        recomendation_size,
        max_patience,
        plot_interval,
        profiles,
        user_ids,
        hyper_params,
        path
    )

  @classmethod
  def _reset_env(
      cls,
      api_client,
      interactions_test_set,
      items
  ):
      for profile in api_client.profiles():
          api_client.remove_interactions_by_user_id(profile.email)
          api_client.delete_profile(profile.email)

      profiles, user_ids = cls._create_fake_profiles(interactions_test_set, items)

      [api_client.add_profile(profile) for profile in profiles]

      return profiles, user_ids

  @classmethod
  def _genres_count(cls, interactions_test_set, items, user_id):
      df = interactions_test_set[interactions_test_set['user_id'] == user_id]

      item_ids = df['item_id'].unique()

      genres = []
      for item_id in item_ids:
          genres.extend(items[items['movie_id'] == item_id]['movie_genres'].tolist()[0])

      return pd.Series(genres).value_counts().reset_index(name='count')


  @classmethod
  def releases(cls, interactions_test_set, items, user_id):
      df = interactions_test_set[interactions_test_set['user_id'] == user_id]

      return np.sort(items[items['movie_id'].isin(df['item_id'].unique())]['movie_release_year'].unique())


  @classmethod
  def _create_fake_profiles(cls, interactions_test_set, items):
      fake = Faker()

      to_email = lambda name: name.lower().replace(' ', '.') + '@gmail.com'

      user_ids = interactions_test_set['user_id'].unique()

      profiles = []
      emails   = []
      profile_user_ids = []
      for user_id in user_ids:
          while True:
              name  = fake.name()
              email = to_email(name)
              if email not in emails:
                  break

          profile_user_ids.append(user_id)
          emails.append(email)

          genres_count_df = cls._genres_count(interactions_test_set, items, user_id=user_id)
          genres = genres_count_df[genres_count_df['count'] >= 10]['index'].tolist()

          release = str(cls.releases(interactions_test_set, items, user_id=user_id)[0])

          profiles.append(client.UserProfileDto(
              name             = name,
              email            = email,
              preferred_from   = release,
              preferred_genres = genres
          ))

      logging.debug(f'Profiles: {len(profiles)}')
      logging.debug(f'Users: {len(user_ids)}')

      return profiles, profile_user_ids

