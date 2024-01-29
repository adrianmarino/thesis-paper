from models import ChatHistory
from models import UserInteraction


class InteractionService:
    def __init__(self, ctx):
        self.ctx = ctx


    async def add(self, user_interaction: UserInteraction):
        await self.ctx.interactions_repository.add_one(user_interaction)
        return user_interaction


    async def find_by_user_id(self, user_id: str):
        return await self.ctx.interactions_repository.find_many_by(user_id=user_id)


    async def find_all(self):
        return await self.ctx.interactions_repository.find_many_by()


    async def delete_one_by(self, user_id, item_id):
        return await self.ctx.interactions_repository.delete_many_by(user_id=user_id, item_id=item_id)

