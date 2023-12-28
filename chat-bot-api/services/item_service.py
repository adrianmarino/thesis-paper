from models import Item


class ItemService:
    def __init__(self, ctx):
        self.ctx = ctx


    async def add(self, item: Item):
        await self.ctx.items_repository.add_one(item)
        self.ctx.items_emb_repository.add(item)
        return await self.find_by_id(item.id)


    async def update(self, item: Item):
        await self.ctx.items_repository.update(item)
        return await self.find_by_id(item.id)


    async def find_by_id(self, item_id: str):
        model = await self.ctx.items_repository.find_by_id(item_id)
        if model is None:
            return model

        emb = self.ctx.items_emb_repository.find_by_id(item_id)
        return model.set_embedding(emb.emb)


    async def delete(self, item_id):
        self.ctx.items_emb_repository.delete(item_id)
        return await self.ctx.items_repository.delete_by_id(item_id)
