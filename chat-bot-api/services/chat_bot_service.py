from models import UserMessage, AIMessage, ChatSession, ChatHistory


class ChatBotService:
  def __init__(self, ctx):
    self.ctx = ctx

  async def send(self, user_message: UserMessage):
    history = await self.ctx.histories_repository.find_by_id(user_message.author)

    response = self.ctx.chat_bot.send(user_message.content, history.as_content_list())

    ai_message = AIMessage(content=response.message, metadata=response.metadata)

    await self.__add_dialog(history, [user_message, ai_message])

    return ai_message

  async def __add_dialog(self, history, dialogue):
    if history == None:
      history = ChatHistory(
        email = user_message.author,
        sessions=[ChatSession(dialogue = dialogue)]
      )
      await self.ctx.histories_repository.add_one(history)
    else:
      history.sessions[0].dialogue.extend(dialogue)
      await self.ctx.histories_repository.update(history)
    
    return history
