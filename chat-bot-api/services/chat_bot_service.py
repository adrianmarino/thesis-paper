from models import UserMessage, AIMessage, ChatSession, ChatHistory, UserInteractionInfo
import util as ut

class ChatBotService:
  def __init__(self, ctx):
    self.ctx = ctx
    self._interactions_count = 20
    self._limit              = 5


  async def send(self, user_message: UserMessage):
    history = await self.ctx.history_service.upsert(user_message.author)

    profile = await self.ctx.profile_service.find(user_message.author)

    interactions_info = await self.ctx.interaction_info_service.find_by_user_id(user_message.author)

    candidates = []
    if len(interactions_info) >= self._interactions_count:
      chat_bot = self.ctx.chat_bot_pool_service.with_candidates
      # Get cantidates where....
    else:
      chat_bot = self.ctx.chat_bot_pool_service.without_candidates

    response = chat_bot.send(
      request      = user_message.content,
      user_profile = str(profile),
      candidates   = candidates,
      limit        = self._limit,
      user_history = UserInteractionInfo.to_str(interactions_info),
      chat_history = history.as_content_list()
    )

    response.metadata.pop('chat_history', None)
    ai_message = AIMessage.from_response(response)

    await self.ctx.history_service.append_dialogue(history, user_message, ai_message)

    return ai_message
