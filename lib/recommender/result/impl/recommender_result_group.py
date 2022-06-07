from ..recommender_result import RecommenderResult


class RecommenderResultGroup(RecommenderResult):
    def __init__(self, recommendations): 
        self.recommendations = recommendations

    def show(self):
        for r in self.recommendations:
            r.show()