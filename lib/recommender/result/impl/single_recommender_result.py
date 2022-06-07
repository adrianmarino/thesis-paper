from ..recommender_result import RecommenderResult


class SingleRecommenderResult(RecommenderResult):
    def __init__(self, name, item, recommendations): 
        self.name = name
        self.item = item
        self.recommendations = recommendations

    def show(self):
        print(f'\nRecommender: {self.name}')
        print(f'Item')
        display(self.item)
        print(f'Recommendations')
        display(self.recommendations)