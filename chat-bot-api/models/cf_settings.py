from .model import Model
from .recommender_settings import RecommenderSettings
import typing


class CFSettings(RecommenderSettings):
    text_query_limit               : int   = 2000,
    not_seen                       : bool  = True,
    k_sim_users                    : int   = 5,
    random_selection_items_by_user : int   = 0.5,
    max_items_by_user              : int   = 5,
    min_rating_by_user             : float = 3.5