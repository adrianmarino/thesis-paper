from .model import Model
import typing


class RecommenderSettings(Model):
    model: str            = 'llama2-7b-chat'
    plain: bool           = False
    metadata: bool        = False
    retry: int            = 2
    shuffle: bool         = False
    limit: int            = 5
    parse_limit: int      = 15
    candidates_limit: int = 50
