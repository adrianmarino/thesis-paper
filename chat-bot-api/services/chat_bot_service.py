from models import UserMessage, AIMessage, ChatSession, ChatHistory
import util as ut

class ChatBotService:
  def __init__(self, ctx):
    self.ctx = ctx


  async def send(self, user_message: UserMessage):
    history = await self.ctx.history_service.upsert(user_message.author)

    profile = await self.ctx.profile_service.find(user_message.author)

    interactions = await self.ctx.interaction_service.find_by_user_id(user_message.author)
    items        = await self.ctx.item_service.find_by_ids([i.item_id for i in interactions])
    user_interactions = [f'Title: {item.title}, Rating: {int.rating}' for int,item in zip(interactions, items)]

    response = self.ctx.chat_bot.send(
      request      = user_message.content,
      user_profile = self.ctx.profile_mapper.to_dict(profile),
      candidates   = [],
      limit        = 5,
      user_history = user_interactions,
      chat_history = history.as_content_list()
    )

    response.metadata.pop('chat_history', None)
    ai_message = AIMessage.from_response(response)

    await self.ctx.history_service.append_dialogue(history, user_message, ai_message)

    return ai_message