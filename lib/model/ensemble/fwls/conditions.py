from abc import ABC
import util as ut


class FeatureWightCondition(ABC):
    def __call__(self, ctx):
        pass
    
class Range(FeatureWightCondition):
    def __init__(self, begin=None, end=None):
        self.__begin = begin
        self.__end   = end

    def __call__(self, ctx):
        if self.__begin and self.__end:
            return int(self.__begin <= ctx.n_interactions <= self.__end)
        elif self.__begin and self.__end is None:
            return int(self.__begin <= ctx.n_interactions)
        elif self.__begin is None and self.__end:
            return int(ctx.n_interactions <= self.__end)
        else:
            raise Exception("Invalid range")

class ConditionContext:
    def __init__(self, user_id, n_interactions):
        self.user_id = user_id
        self.n_interactions = n_interactions
        

class ConditionContextFactory:
    def __init__(self, user_ids):
        self.user_id_times = ut.value_counts(user_ids)

    def create(self, user_id): return ConditionContext(user_id, self.user_id_times[user_id.item()])