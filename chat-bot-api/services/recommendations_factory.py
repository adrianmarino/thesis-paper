from models import UserMessage, AIMessage, ChatSession, ChatHistory, UserInteractionInfo, Recommendation
import util as ut
import pandas as pd

class RecommendationsFactory:
  def __init__(self, ctx):
    self.ctx = ctx


  async def create(self, raw_recommendations, email, base_url):
    recommendations = []

    for r in list(raw_recommendations):
      sim_items, distances = await self.ctx.item_service.find_by_title(r['title'], limit=1)

      if distances[0] >= 0 and distances[0] <= 1:
        item = sim_items[0]
        recommendations.append(Recommendation(
          title       = r['title'] + f' (Sim: {item.title.strip()})',
          release     = r['release'],
          description = r['description'],
          rating      = r['rating'],
          votes       = [ f'{base_url}api/v1/interactions/make/{email}/{item.id}/{i}' for i in range(1, 6)]
        ))

    return recommendations