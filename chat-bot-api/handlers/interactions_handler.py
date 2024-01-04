from models import UserInteraction
from fastapi import HTTPException, APIRouter, Response
from repository.mongo import EntityAlreadyExistsException
import util as ut


def interactions_handler(base_url, ctx):
    router = APIRouter(prefix=f'{base_url}/interactions')


    @router.post('')
    async def add_interaction(user_interaction: UserInteraction):
        try:
            return await ctx.interaction_service.add(user_interaction)
        except EntityAlreadyExistsException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Already exist this interaction. Cause: {e}"
            )

    @router.get('/make/{user_id}/{item_id}/{rating}', status_code=204)
    async def make_interaction(user_id: str, item_id: str, rating: int):
        try:
            user_interaction = UserInteraction(user_id=user_id, item_id=item_id, rating=rating)
            await ctx.interaction_service.add(user_interaction)
        except EntityAlreadyExistsException as e:
            return Response(status_code=204)


    @router.get('/{user_id}', status_code = 200)
    async def get_interactions_by_user_id(user_id: str):
        interactions = await ctx.interaction_service.find_by_user_id(user_id)

        if ut.empty(interactions):
            raise HTTPException(status_code=404, detail=f'Not found interactions for {user_id} profile')
        else:
            return [i.dict(exclude_none=True) for i in interactions]


    @router.delete("/{user_id}/{item_id}", status_code = 202)
    async def delete_interactions(user_id: str, item_id: str):
        await ctx.interaction_service.delete_one_by(user_id=user_id, item_id=item_id)
        return Response(status_code=202)


    return router