from models import UserMessage, AIMessage, ChatSession, ChatHistory


class ChatBotService:
  def __init__(self, ctx):
    self.ctx = ctx

  async def send(self, user_message: UserMessage):
    history = await self.ctx.history_service.upsert(user_message.author)

    response = self.ctx.chat_bot.send(user_message.content, history.as_content_list())

    ai_message = AIMessage.from_response(response)

    await self.ctx.history_service.append_dialogue(history, user_message, ai_message)

    return ai_message