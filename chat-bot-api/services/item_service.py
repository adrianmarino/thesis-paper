from models import Item


class ItemService:
    def __init__(self, ctx):
        self.ctx = ctx


    async def add(self, item: Item):
        await self.ctx.items_repository.add_one(item)
        return await self.ctx.items_repository.find_by_id(item.id)


    async def update(self, item: Item):
        await self.ctx.items_repository.update(item)
        return await self.ctx.items_repository.find_by_id(item.id)


    async def find_by_id(self, item_id: str):
        return await self.ctx.items_repository.find_by_id(item_id)


    async def delete(self, item_id):
        return await self.ctx.items_repository.delete_by_id(item_id)
