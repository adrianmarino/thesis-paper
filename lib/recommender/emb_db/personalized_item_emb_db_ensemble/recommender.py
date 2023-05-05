import pandas as pd


from recommender import PersonalizedItemRecommender, PersonalizedItemEmbDBEnsembleRecommenderResult


def group_mean(df, group_col, mean_col):
    return df.groupby([group_col])[mean_col].mean().reset_index()


class PersonalizedItemEmbDBEnsembleRecommender(PersonalizedItemRecommender):

    def __init__(
            self,
            recommenders,
            weights,
            recommender_k=100
    ):
        self.recommenders = recommenders
        self.recommender_k = recommender_k
        self.weights = weights


    def __resolve_recommendations(self, user_id):
        results = []
        for idx, rec in enumerate(self.recommenders):
            result = rec.recommend(user_id=user_id, k=self.recommender_k).data
            result['recommender'] = idx
            results.append(result)
        return pd.concat(results)

    def recommend(self, user_id: int, k: int = 5):
        recommendations = self.__resolve_recommendations(user_id)

        recommendations = recommendations[recommendations['sim'] > 0.0]

        recommendations['sim'] = recommendations['sim'] + recommendations['recommender'].apply(
            lambda rec: self.weights[rec])

        recommendations = recommendations \
            .groupby(['sim_id', 'sim_imdb_id', 'sim_title', 'sim_rating'])['sim'] \
            .mean() \
            .reset_index()

        return PersonalizedItemEmbDBEnsembleRecommenderResult(self.name, recommendations, k)


    @property
    def name(self):
        return f'Ensemble of {", ".join([r.name for r in self.recommenders])}.'