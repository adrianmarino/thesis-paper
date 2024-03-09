from .model import Model
import typing


class RecommenderSettings(Model):
    shuffle                           : bool = False
    candidates_limit                  : int  = 20
    llm_response_limit                : int  = 20
    recommendations_limit             : int  = 5
    similar_items_augmentation_limit  : int  = 5