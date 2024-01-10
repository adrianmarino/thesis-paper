from models import Recommendation, Recommendations
import util as ut
import pandas as pd
import random


def cosine_similarity(text1, text2):
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform([text1, text2])
  return cosine_similarity(X[0], X[1])[0][0]


class RecommendationsFactory:
  def __init__(self, ctx):
    self.ctx = ctx


  async def create(self, response, email, base_url, limit, include_metadata=False):
    recommended_items = []

    for r in response.metadata['recommendations']:
      sim_items, distances = await self.ctx.item_service.find_by_content(r['title'], limit=1)
      item = sim_items[0]

      title_sim = cosine_similarity(r['title'], item.title.strip())

      if distances[0] > 0 and title_sim >= 0.1:
        metadata = None
        if include_metadata:
          metadata      = {
            'position' : r['number'],
            'db_item': {
              'title'     : item.title.strip(),
              'rating'    : item.rating,
              'query_sim' : 1 - distances[0],
              'title_sim' : title_sim
            }
          }

        recommended_items.append(
          Recommendation(
            title         = r['title'],
            poster        = None if item.poster == 'None' else item.poster,
            release       = r['release'],
            description   = r['description'],
            genres        = item.genres,
            votes         = [ f'{base_url}api/v1/interactions/make/{email}/{item.id}/{i}' for i in range(1, 6)],
            metadata      = metadata
          ).dict(exclude_none=True)
        )

    if len(recommended_items) > limit:
      recommended_items = random.sample(recommended_items, limit)

    return Recommendations(
      items=recommended_items,
      response=response
    )
