

class NotFoundRecommenderException(Exception):
    pass


class NotFoundSimilarUsersException(NotFoundRecommenderException):
    def __init__(self, user_id, k_sim_users):
        self.message = f'Not found {k_sim_users} similar users. Perhaps it is necessary to retrain collaborative filtering models to include {user_id} in the user embeddings latent space. This step is essential as it enables the search for {user_id} {k_sim_users} nearest neighboring users with similar behaviors and preferences.'



class NotFoundSimilarUserInteractionsException(NotFoundRecommenderException):
    def __init__(
        self,
        similar_user_ids,
        max_items_by_user,
        min_rating_by_user,
    ):
        self.message = f'Not found interactions for k similar users: UserIds: {similar_user_ids}. MaxItemsByUser: {max_items_by_user}, MinRatingByUser: {min_rating_by_user}.'



class NotFoundSimilarInteractionItemsException(NotFoundRecommenderException):
    def __init__(self):
        self.message = f'Not found items from similar users interactions.'



class NotFoundItemByTextQueryException(NotFoundRecommenderException):
    def __init__(self, prompt, limit):
        self.message = f'Not found items by "{prompt}" with {limit} limit.'