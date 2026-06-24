from .recommender_settings import RecommenderSettings
from pydantic import Field

class CFSettings(RecommenderSettings):
    text_query_limit               : int   = Field(5000, description="Character limit for the textual search in collaborative filtering.")
    not_seen                       : bool  = Field(True, description="If True, excludes movies that the user has already rated from the recommendations.")
    k_sim_users                    : int   = Field(20, description="Number of similar users (nearest neighbors) to consider when calculating predictions.")
    random_selection_items_by_user : float = Field(1.0, description="Percentage of randomly selected items from the history of similar users (adds diversity).")
    max_items_by_user              : int   = Field(30, description="Limit of items to extract from the history of each similar user.")
    min_rating_by_user             : float = Field(3.5, description="Minimum rating an item must have in a similar user's history to be considered a good candidate.")
    rank_criterion                 : str   = Field('user_sim_weighted_pred_rating_score', description="Candidate re-ranking criterion. Options: user_sim_weighted_rating_score, user_item_sim, pred_user_rating")
    neighborhood_expansion_ratio   : float = Field(1.5, description="Multiplier ratio to dynamically expand the search space (k_sim_users and max_items_by_user) if the initial CF pass yields too few unwatched candidates.")
    max_expansion_attempts         : int   = Field(3, description="Maximum number of expansion loop iterations allowed to prevent infinite database queries.")
