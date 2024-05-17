from models import Item
from fastapi import HTTPException, APIRouter, Response
from repository.mongo import EntityAlreadyExistsException
import util as ut
from services import ItemSimQuery


def items_handler(base_url, ctx):
    router = APIRouter(prefix=f'{base_url}/items')


    @router.post('', status_code=201)
    async def add_item(item: Item):
        try:
            return await ctx.item_service.add_one(item)
        except EntityAlreadyExistsException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Already exist this item. Cause: {e}"
            )


    @router.post('/bulk', status_code=201)
    async def add_items(items: list[Item]):
        try:
            await ctx.item_service.add_many(items)
            return Response(status_code=201)
        except EntityAlreadyExistsException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Already exist any item. Cause: {e}"
            )


    @router.get('/{item_id}', status_code = 200)
    async def get_item(item_id: str, hide_emb: bool = True):
        item = await ctx.item_service.find_by_id(item_id)

        if item is None:
            raise HTTPException(status_code=404, detail=f'Not found {item_id} item')
        else:
            return remove_embedding([item], hide_emb)[0]


    @router.get('', status_code = 200)
    async def get_item(
        email    : str | None = None,
        seen     : bool       = True,
        content  : str | None = None,
        all      : bool       = False,
        limit    : int        = 5,
        hide_emb : bool       = True,
        release  : int        = 1950,
        rating   : float      = 0.0
    ):
        if all:
            return remove_embedding(await ctx.item_service.find_all(not hide_emb), hide_emb)
        elif email and content is None:
            if seen:
                return remove_embedding(await ctx.item_service.find_by_user_id(email), hide_emb)
            else:
                return remove_embedding(await ctx.item_service.find_unseen_by_user_id(email, limit), hide_emb)
        elif content:
            query = ItemSimQuery() \
                    .user_id_eq(email) \
                    .is_seen(seen) \
                    .rating_gte(rating) \
                    .contains(content) \
                    .release_gte(release) \
                    .limit_eq(limit)

            items, distances = await ctx.item_service.find_similars_by(query)
            return { 'items': remove_embedding(items, hide_emb), 'distances': distances}
        else:
            raise HTTPException(status_code=400, detail=f'Missing filter params: all | email | content')


    @router.put('/embeddings/content/build', status_code = 202)
    async def build_content_embeddings(batch_size: int =100):
        return await ctx.item_service.rebuild_content_embeddings(batch_size)


    @router.delete("/{item_id}", status_code = 202)
    async def delete_item(item_id: str):
        pu.set_device_name('gpu')
        await ctx.item_service.delete(item_id)
        return Response(status_code=202)


    return router


def remove_embedding(items, hide_emb):
    result = []
    for item in items:
        if hide_emb:
            item.embedding = None
        result.append(item.model_dump(exclude_none=True))
    return result
