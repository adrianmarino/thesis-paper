import os
from bunch import Bunch
import logging
import util as ut


class InteractionsChangeDetector:
    def __init__(self,
        state_path               : str,
        user_seq_col             : str,
        item_seq_col             : str,
        update_period_in_minutes : int = 180, # 3 hours
    ):
        self._state_path             = state_path
        self._user_seq_col           = user_seq_col
        self._item_seq_col           = item_seq_col
        self._update_period_in_minutes = update_period_in_minutes


    def _load_state(self):
        if os.path.exists(self._state_path):
            return Bunch(ut.Picket.load(self._state_path))
        else:
            return Bunch({})


    def _n_users(self, df):
        return df[self._user_seq_col].unique().shape[0]


    def _n_items(self, df):
        return df[self._item_seq_col].unique().shape[0]


    def _n_interactions(self, df):
        return df.shape[0]


    def update(self, df):
        new_state = {
            'n_users'  : self._n_users(df),
            'n_items'  : self._n_items(df),
            'n_interactions': self._n_interactions(df),
            'datetime' : ut.DateTimeUtils.now()
        }
        ut.Picket.save(self._state_path, new_state)


    def detect(self, df)-> bool:
        state = self._load_state()
        if 'n_users' not in state:
            logging.info(f'Not found an interactions change registers. Assume that users/items/interactions change from nothing.')
            return True

        diff = ut.DateTimeUtils.diff_with_now(state.datetime)

        if diff.minutes >= self._update_period_in_minutes:
            logging.info(f'{self._update_period_in_minutes} minutes wait time expired!.')

            if state.n_users != self._n_users(df):
                logging.info(f'Found {self._n_users(df) - state.n_users} new users.')
                return True
            else:
                logging.info(f'Not found users count changes.')

            if state.n_items != self._n_items(df):
                logging.info(f'Found {self._n_items(df) - state.n_items} new items.')
                return True
            else:
                logging.info(f'Not found items count changes.')

            if self._n_interactions(df) != state.n_interactions:
                logging.info(f'Found that interactions count change from {state.n_interactions} to {self._n_interactions(df)} after {self._update_period_in_minutes} hours.')
                return True
            else:
                logging.info(f'Not found interactions count changes.')

            self.update(df)
        else:
            logging.info(f'Waiting {self._update_period_in_minutes - diff.minutes} minutes to change interactions.')

        return False
