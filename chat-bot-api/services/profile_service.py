from models import ChatHistory
from models import UserProfile


class ProfileService:
    def __init__(self, ctx):
        self.ctx = ctx


    async def add(self, user_profile: UserProfile):
        await self.ctx.profiles_repository.add_one(user_profile)
        return await self.ctx.profiles_repository.find_by_id(user_profile.email)


    async def update(self, user_profile: UserProfile):
        await self.ctx.profiles_repository.update(user_profile)
        return await self.ctx.profiles_repository.find_by_id(user_profile.email)


    async def find(self, email: str):
        return await self.ctx.profiles_repository.find_by_id(email)


    async def all(self):
        return await self.ctx.profiles_repository.find_all()


    async def delete(self, email: str):
        return await self.ctx.profiles_repository.delete_by_id(email)
