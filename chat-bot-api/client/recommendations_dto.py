from .recommendation_dto import RecommendationDto

class RecommendationsDto:
    def __init__(self, response, verbose):
        self.items = [RecommendationDto(item, verbose) for item in response['items']]
        self.response = response