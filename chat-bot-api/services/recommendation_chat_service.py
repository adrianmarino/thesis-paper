from models import UserMessage, AIMessage, ChatHistory, UserInteractionInfo, LangChainMessageMapper
import util as ut
import pandas as pd
import sys
import logging
import pytorch_common.util as pu
from .item_sim_query import ItemSimQuery
from models import RecommendationQuery


class RecommendationChatService:
  def __init__(self, ctx):
    self.ctx = ctx
    self._interactions_count = 20


  def available_models(self):
    return self.ctx.chat_bot_pool_service.available_models()


  async def ask(self, query: RecommendationQuery):
    while True:
      recommendations = await self.__query(query)

      if not recommendations.empty or query.settings.retry == 0:
        break

      query.settings.retry -= 1

    return recommendations

  async def __query(self, query: RecommendationQuery):
    interactions_info = await self.ctx.interaction_info_service.find_by_user_id(
      query.message.author
    )

    prompt = f'prompt{int(len(interactions_info) >= self._interactions_count)}'

    chat_bot = self.ctx.chat_bot_pool_service.get(
      model  = query.settings.llm,
      prompt = prompt
    )

    profile = await self.ctx.profile_service.find(query.message.author)

    candidate_items, logs = await self.__find_candidates(query, profile, interactions_info)

    history = await self.ctx.history_service.upsert(query.message.author)
    # chat_history = LangChainMessageMapper().to_lang_chain_messages(history.dialogue)[-2:]
    chat_history = []

    seen_items = [info.item for info in interactions_info]

    rec_settings = query.settings.collaborative_filtering if self.__has_min_interactions(interactions_info) else query.settings.rag

    sw = pu.Stopwatch()

    logs.append(f'Start inference - LLM: {query.settings.llm}. Prompt: {prompt}')
    logging.info(logs[-1])
    response = chat_bot(
      request      = f'Question:\n {query.message.content}',
      user_profile = str(profile),
      candidates   = self.__items_to_str_list(candidate_items, 'Candidate movies (with rating)'),
      limit        = rec_settings.llm_response_limit,
      user_history = self.__items_to_str_list(
        seen_items,
        'Seen movies (with rating)',
        'The user has not seen any movie at the moment.'
      ),
      chat_history = chat_history
    )
    logs.append(f'End {query.settings.llm} llm inference. Elapsed time: {sw.to_str()}.')
    logging.info(logs[-1])

    ai_message = AIMessage.from_response(response, query.message.author)

    await self.ctx.history_service.append_dialogue(history, query.message, ai_message)

    result = await self.ctx.recommendations_factory.create(
      response,
      query.message.author,
      query.settings.base_url,
      limit                            = rec_settings.recommendations_limit,
      include_metadata                 = query.settings.include_metadata,
      shuffle                          = rec_settings.shuffle,
      similar_items_augmentation_limit = rec_settings.similar_items_augmentation_limit
    )


    if query.settings.include_metadata:
      result.metadata['elapsed_time'] = sw.to_str()
      result.metadata['logs'] = logs
    else:
      result.metadata = None

    return result


  def __items_to_str_list(self, items, title, fallback=''):
    if len(items) > 0:
      str_items = [f'- {item.title.strip()}: {item.rating:.1f}' for item in items]

      return f'{title}:\n' + '\n'.join(str_items)
    else:
      return f'{fallback}\n'


  def __has_min_interactions(self, interactions_info):
    return len(interactions_info) >= self._interactions_count


  async def __find_candidates(
    self,
    query,
    profile,
    interactions_info
  ):
    logs = []
    if self.__has_min_interactions(interactions_info):
      recommendations = await self.ctx.database_user_item_filtering_recommender.recommend(
          user_id            = query.message.author,
          text_query         = query.message.content,
          k_sim_users        = query.settings.collaborative_filtering.k_sim_users,
          max_items_by_user  = query.settings.collaborative_filtering.max_items_by_user,
          text_query_limit   = query.settings.collaborative_filtering.text_query_limit,
          min_rating_by_user = query.settings.collaborative_filtering.min_rating_by_user,
          not_seen           = query.settings.collaborative_filtering.not_seen
      )

      ordered_recommendations = recommendations \
        .data \
        .sort_values(
          by        = [query.settings.collaborative_filtering.rank_criterion],
          ascending = False,
        )

      if len(ordered_recommendations['id']) > 0:
        item_ids = ordered_recommendations['id'] \
          .unique() \
          .tolist()[:query.settings.collaborative_filtering.candidates_limit]
      else:
        item_ids = []

      candidate_items = await self.ctx.item_service.find_by_ids(item_ids)

      logs.append(f'Candidates items source: Collaborative Filtering Recommender')
      logging.info(logs[-1])

    else:
      sim_items_query = ItemSimQuery() \
          .user_id_eq(query.message.author) \
          .contains(query.message.content) \
          .is_seen(query.settings.rag.not_seen) \
          .release_gte(profile.release_from) \
          .limit_eq(query.settings.rag.candidates_limit)

      candidate_items, _ = await self.ctx.item_service.find_similars_by(sim_items_query)

      candidate_items = candidate_items[:query.settings.rag.candidates_limit]

      logs.append(f'Candidates items source: RAG Query')
      logging.info(logs[-1])

    logs.append(f'Found {len(candidate_items)} candidate items')
    logging.info(logs[-1])

    return candidate_items, logs
