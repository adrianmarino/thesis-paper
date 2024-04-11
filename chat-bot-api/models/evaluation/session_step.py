import metric as mt
import util as ut
import numpy as np


@ut.printable
class SessionStep:
    def __init__(self, data): self.data = data

    @property
    def recommended_item_ids(self): return [int(id) for id in self.data['recommended_items']]

    @property
    def relevant_item_ids(self): return [int(id) for id in self.data['relevant_item_ratings'].keys()]

    @property
    def recall(self): return np.array(mt.recall(self.recommended_item_ids, self.relevant_item_ids))

    def relevante_rating(self, item_id): return self.data['relevant_item_ratings'][item_id]

    def _state(self): return self.data

    @property
    def ndcg(self):
        ordered_user_asigned_ratings = []
        for item_id in self.recommended_item_ids:
            if item_id in self.relevant_item_ids:
                ordered_user_asigned_ratings.append(self.relevante_rating(item_id))

        return mt.ndcg(ordered_user_asigned_ratings)