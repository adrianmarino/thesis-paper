from models import UserMessage, AIMessage, ChatSession, ChatHistory, UserInteractionInfo
import util as ut
import pandas as pd


class ChatBotService:
  def __init__(self, ctx):
    self.ctx = ctx
    self._interactions_count = 20
    self._limit = 5


  async def send(
    self,
    user_message: UserMessage,
    model: str = 'ollama2',
    base_url='',
    include_metadata=False
  ):
    history = await self.ctx.history_service.upsert(user_message.author)

    profile = await self.ctx.profile_service.find(user_message.author)

    interactions_info = await self.ctx.interaction_info_service.find_by_user_id(user_message.author)

    candidates = []
    if len(interactions_info) >= self._interactions_count:
      chat_bot = self.ctx.chat_bot_pool_service.get(model, with_candidates=True)
      # Get candidates where....
    else:
      chat_bot = self.ctx.chat_bot_pool_service.get(model, with_candidates=False)

    response = chat_bot.send(
      request      = user_message.content,
      user_profile = str(profile),
      candidates   = candidates,
      limit        = 10,
      user_history = self.__str_user_history(interactions_info),
      chat_history = history.as_content_list()[-10:]
    )

    ai_message = AIMessage.from_response(response)

    await self.ctx.history_service.append_dialogue(history, user_message, ai_message)

    return await self.ctx.recommendations_factory.create(
      response,
      user_message.author,
      base_url,
      include_metadata
    )


  def __str_user_history(self, interactions_info):
    movies = [f'- {info.title.strip()}: {info.rating}' for info in interactions_info]

    return 'Seen movies (with rating):\n' + '\n'.join(movies)