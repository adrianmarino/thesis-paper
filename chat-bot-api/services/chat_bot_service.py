from models import UserMessage, AIMessage, ChatSession, ChatHistory
import util as ut

class ChatBotService:
  def __init__(self, ctx):
    self.ctx = ctx


  async def send(self, user_message: UserMessage):
    history = await self.ctx.history_service.upsert(user_message.author)

    profile = await self.ctx.profile_service.find(user_message.author)

    user_history = [] # interactions

    response = self.ctx.chat_bot.send(
      request      = user_message.content,
      user_profile = self.ctx.profile_mapper.to_dict(profile),
      candidates   = [],
      limit        = 5,
      user_history = user_history,
      chat_history = history.as_content_list()
    )

    response.metadata.pop('chat_history', None)
    ai_message = AIMessage.from_response(response)

    await self.ctx.history_service.append_dialogue(history, user_message, ai_message)

    return ai_message