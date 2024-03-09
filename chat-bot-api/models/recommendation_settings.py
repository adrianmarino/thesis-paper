from .model import Model
from .cf_settings import CFSettings
from .rag_settings import RagSettings
import typing


class RecommendationSettings(Model):
    model                             : str  = 'llama2-7b-chat'
    retry                             : int  = 2
    base_url                          : str  = ""
    plain                             : bool = False
    include_metadata                  : bool = False
    collaborative_filtering           : CFSettings
    rag                               : RagSettings

    class Config:
        arbitrary_types_allowed = True