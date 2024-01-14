from models import Recommendation, Recommendations
import util as ut
import pandas as pd
import random
import sys
import logging


def cosine_similarity(text1, text2):
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform([text1, text2])
  return cosine_similarity(X[0], X[1])[0][0]


class RecommendationsFactory:
  def __init__(self, ctx):
    self.ctx = ctx


  async def create(self, response, email, base_url, limit, include_metadata=False, shuffle=False):
    recommended_item_by_title = {}
    excluded_recommended_items = []

    for r in response.metadata['recommendations']:
      sim_items, distances = await self.ctx.item_service.find_by_content(r['title'], limit=5)

      for idx, item in enumerate(sim_items):
        query_sim   = 1 - distances[idx]
        title_sim   = cosine_similarity(r['title'], item.title.strip())
        release_sim = cosine_similarity(r['release'], item.release.strip())

        recommendation = self._create_recommendation(
          r,
          item,
          include_metadata,
          query_sim,
          title_sim,
          release_sim,
          base_url,
          email
        )

        title_is_more_related = (title_sim >= 0.9 and release_sim <= 0.3)

        if abs(query_sim) > 0 and (title_is_more_related or (title_sim >= 0.1 and release_sim >= 0.3)) and recommendation.title not in recommended_item_by_title:
          recommended_item_by_title[recommendation.title] = recommendation
        else:
          excluded_recommended_items.append(recommendation)


    recommended_items = list(recommended_item_by_title.values())

    if shuffle:
      recommended_items = random.sample(recommended_items, len(recommended_items))

    if len(recommended_items) > limit:
      recommended_items = recommended_items[:limit]


    return Recommendations(
      items    = recommended_items,
      metadata = {
        'excluded_items': excluded_recommended_items,
        'response': response
      }
    )


  def _create_recommendation(
    self,
    rec_item,
    found_item,
    include_metadata,
    query_sim,
    title_sim,
    release_sim,
    base_url,
    email
  ):
      metadata = None
      if include_metadata:
        metadata      = {
          'result_item': {
            'position' : rec_item['number'],
            'title'    : rec_item['title']
          },
          'db_item': {
            'id'          : found_item.id,
            'release'     : found_item.release,
            'rating'      : found_item.rating,
            'query_sim'   : query_sim,
            'title'       : found_item.title.strip(),
            'title_sim'   : title_sim,
            'release_sim' : release_sim
          }
        }

      return Recommendation(
        title         = found_item.title.strip(),
        poster        = None if found_item.poster == 'None' else found_item.poster,
        release       = rec_item['release'],
        description   = rec_item['description'],
        genres        = found_item.genres,
        votes         = [ f'{base_url}api/v1/interactions/make/{email}/{found_item.id}/{i}' for i in range(1, 6)],
        metadata      = metadata
      )
