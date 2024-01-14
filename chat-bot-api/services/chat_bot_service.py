from models import UserMessage, AIMessage, ChatHistory, UserInteractionInfo, LangChainMessageMapper
import util as ut
import pandas as pd
import sys
import logging
import pytorch_common.util as pu


class ChatBotService:
  def __init__(self, ctx):
    self.ctx = ctx
    self._interactions_count = 20


  def available_models(self):
    return self.ctx.chat_bot_pool_service.available_models()


  async def send(
    self,
    user_message: UserMessage,
    model: str,
    base_url         = '',
    include_metadata = False,
    shuffle          = False,
    candidates_limit = 50,
    parse_limit      = 15,
    result_limit     = 5
  ):
    history = await self.ctx.history_service.upsert(user_message.author)

    profile = await self.ctx.profile_service.find(user_message.author)

    interactions_info = await self.ctx.interaction_info_service.find_by_user_id(user_message.author)
    seen_items = [info.item for info in interactions_info]

    if len(interactions_info) >= self._interactions_count:
      chat_bot = self.ctx.chat_bot_pool_service.get(model, with_candidates=True)
      # Get candidates from a recommendation model.
      candidate_items = []
    else:
      candidate_items,_ = await self.ctx.item_service.find_unseen_by_content(
        user_id = user_message.author,
        content = user_message.content,
        release_from = profile.release_from,
        genres = profile.genres,
        limit   = candidates_limit
      )
      chat_bot = self.ctx.chat_bot_pool_service.get(model, with_candidates=False)

    chat_history = LangChainMessageMapper().to_lang_chain_messages(history.dialogue)[-2:]

    sw = pu.Stopwatch()
    logging.info(f'Start inference with {model} model')
    response = chat_bot.send(
      request      = user_message.content,
      user_profile = str(profile),
      candidates   = self.__items_to_str_list(candidate_items, 'Candidate movies (with rating)'),
      limit        = parse_limit,
      user_history = self.__items_to_str_list(
        seen_items,
        'Seen movies (with rating)',
        'The user has not seen any movie at the moment.'
      ),
      chat_history = chat_history
    )
    logging.info(f'End {model} model inference. Elapsed time: {sw.to_str()}.')

    ai_message = AIMessage.from_response(response, user_message.author)

    await self.ctx.history_service.append_dialogue(history, user_message, ai_message)

    result = await self.ctx.recommendations_factory.create(
      response,
      user_message.author,
      base_url,
      limit            = result_limit,
      include_metadata = include_metadata,
      shuffle          = shuffle
    )

    result.metadata['elapsed_time'] = sw.to_str()

    return result


  def __items_to_str_list(self, items, title, fallback=''):
    if len(items) > 0:
      str_items = [f'- {item.title.strip()}: {item.rating}' for item in items]

      return f'{title}:\n' + '\n'.join(str_items)
    else:
      return f'{fallback}\n'