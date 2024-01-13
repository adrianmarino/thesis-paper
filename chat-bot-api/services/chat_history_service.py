from models import ChatHistory
import sys

class ChatHistoryService:
    def __init__(self, ctx):
        self.ctx = ctx


    async def upsert(self, email):
        history = await self.ctx.histories_repository.find_by_id(email)

        if history == None:
            history = ChatHistory(email = email, dialogue = [])
            await self.ctx.histories_repository.add_one(history)

        return history


    async def append_dialogue(self, history, user_message, ai_message):
        history.append_dialogue(user_message, ai_message)
        await self.ctx.histories_repository.update(history)
        return history


    async def find(self, email):
        return await self.ctx.histories_repository.find_by_id(email)


    async def delete_by_id(self, email):
        return await self.ctx.histories_repository.delete_by_id(email)
