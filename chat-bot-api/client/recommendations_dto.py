from .recommendation_dto import RecommendationDto

class RecommendationsDto:
    def __init__(self, response):
        self.items = [RecommendationDto(item) for item in response['items']]
        self.response = response