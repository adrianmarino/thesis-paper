from models import UserInteraction
from fastapi import HTTPException, APIRouter, Response
from repository.mongo import EntityAlreadyExistsException
import util as ut


def interactions_handler(base_url, ctx):
    router = APIRouter(prefix=f'{base_url}/interactions', tags=["Interactions"])


    @router.post('', summary="Add a Single Interaction (Rating)")
    async def add_interaction(user_interaction: UserInteraction):
        """
        Records an explicit feedback (rating) given by a user to a specific movie.
        **Note:** This data feeds the Collaborative Filtering Airflow DAGs that regularly re-train the models.
        """
        try:
            return await ctx.interaction_service.add_one(user_interaction)
        except EntityAlreadyExistsException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Already exist this interaction. Cause: {e}"
            )


    @router.post('/bulk', status_code=201, summary="Add Multiple Interactions in Bulk")
    async def add_interactions(user_interactions: list[UserInteraction]):
        """
        Efficiently inserts multiple interactions (ratings) at once. Useful for migrating or importing datasets.
        """
        try:
            await ctx.interaction_service.add_many(user_interactions)
            return Response(status_code=201)
        except EntityAlreadyExistsException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Already exist any interaction. Cause: {e}"
            )


    @router.get('/make/{user_id}/{item_id}/{rating}', status_code=204, summary="Quick Add Interaction (GET method)")
    async def make_interaction(user_id: str, item_id: str, rating: int):
        """
        A quick way to record a rating via a simple GET request (e.g., clicking a link).
        """
        try:
            user_interaction = UserInteraction(user_id=user_id, item_id=item_id, rating=rating)
            await ctx.interaction_service.add_one(user_interaction)
        except EntityAlreadyExistsException as e:
            return Response(status_code=204)


    @router.get('/users/{user_id}', status_code = 200, summary="Get Interactions by User ID")
    async def get_interactions_by_user_id(user_id: str | None = None):
        """
        Retrieves all the ratings/interactions a specific user has submitted.
        """
        interactions = await ctx.interaction_service.find_by_user_id(user_id)

        if ut.empty(interactions):
            raise HTTPException(status_code=404, detail=f'Not found interactions for {user_id} profile')
        else:
            return [i.model_dump(exclude_none=True) for i in interactions]


    @router.get('', status_code = 200, summary="Get All Interactions")
    async def get_all_interactions(user_id: str | None = None):
        """
        Retrieves all interactions currently stored in the system.
        """
        interactions = await ctx.interaction_service.find_all()

        if ut.empty(interactions):
            raise HTTPException(status_code=404, detail=f'Not found interactions')
        else:
            return [i.model_dump(exclude_none=True) for i in interactions]


    @router.delete("/users/{user_id}/items/{item_id}", status_code = 202, summary="Delete a specific User-Item Interaction")
    async def delete_interaction(user_id: str, item_id: str):
        """
        Removes a specific rating given by a user to a specific item.
        """
        await ctx.interaction_service.delete_one_by(user_id=user_id, item_id=item_id)
        return Response(status_code=202)


    @router.delete("/users/{user_id}", status_code = 202, summary="Delete All Interactions for a User")
    async def delete_all_user_interactions(user_id: str):
        """
        Purges all ratings submitted by a specific user ID.
        """
        await ctx.interaction_service.delete_all_by_user_id(user_id)
        return Response(status_code=202)


    return router
