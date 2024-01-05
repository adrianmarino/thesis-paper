from models import Item
import util as ut


class ItemService:
    def __init__(self, ctx):
        self.ctx = ctx


    async def add_many(self, items: list[Item]):
        await self.ctx.items_repository.add_many(items)
        self.ctx.items_emb_repository.add_many(items)


    async def add_one(self, item: Item):
        await self.ctx.items_repository.add_one(item)
        self.ctx.items_emb_repository.add_one(item)
        return await self.find_by_id(item.id)


    async def update(self, item: Item):
        await self.ctx.items_repository.update(item)
        return await self.find_by_id(item.id)


    async def find_by_id(self, id: str):
        models = await self.find_by_ids([id])
        return None if ut.empty(models) else models[0]


    async def find_by_ids(self, ids: list[str]):
        models = await self.ctx.items_repository.find_many_by(item_id={ '$in': ids})
        return self._populate_embeddings(models)


    async def find_all(self):
        models = await self.ctx.items_repository.find_many_by()
        return self._populate_embeddings(models)


    async def find_by_user_id(self, user_id: str):
        interactions = await self.ctx.interactions_repository.find_many_by(user_id=user_id)
        items = await self.find_by_ids([i.item_id for i in interactions])
        return self._populate_embeddings(items)


    async def find_by_unseen_by_user_id(self, user_id: str, limit: int=10):
        interactions = await self.ctx.interactions_repository.find_many_by(user_id=user_id)
        items = await self.ctx.items_repository.find_many_by(limit=limit, item_id={ '$nin': [str(i.item_id) for i in interactions]})
        return self._populate_embeddings(items)


    async def find_by_title(self, title: str, limit=5):
        embeddings = self.ctx.emb_service.embeddings([title])
        result = self.ctx.items_emb_repository.search_sims(embeddings, limit)
        items = await self.find_by_ids([str(id) for id in result.ids])
        return self._populate_embeddings(items), result.distances


    async def delete(self, item_id):
        self.ctx.items_emb_repository.delete(item_id)
        return await self.ctx.items_repository.delete_by_id(item_id)


    def _populate_embeddings(self, models):
        results = []
        for m in models:
            emb = self.ctx.items_emb_repository.find_by_id(m.id)
            if emb:
                results.append(m.with_embedding(emb.emb))
        return results