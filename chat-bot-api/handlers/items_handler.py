from models import Item
from fastapi import HTTPException, APIRouter, Response
from repository.mongo import EntityAlreadyExistsException
import util as ut


def items_handler(base_url, ctx):
    router = APIRouter(prefix=f'{base_url}/items')


    @router.post('')
    async def add_item(item: Item):
        try:
            return await ctx.item_service.add(item)
        except EntityAlreadyExistsException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Already exist this item. Cause: {e}"
            )


    @router.get('/{item_id}', status_code = 200)
    async def get_item(item_id: str):
        item = await ctx.item_service.find_by_id(item_id)

        if item is None:
            raise HTTPException(status_code=404, detail=f'Not found {item_id} item')
        else:
            return item


    @router.get('', status_code = 200)
    async def get_item(email: str):
        return await ctx.item_service.find_by_user_id(email)


    @router.delete("/{item_id}", status_code = 202)
    async def delete_item(item_id: str):
        await ctx.item_service.delete(item_id)
        return Response(status_code=202)


    return router