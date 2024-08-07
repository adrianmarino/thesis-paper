from models import UserProfile
from fastapi import HTTPException, APIRouter, Response
from repository.mongo import EntityAlreadyExistsException
import logging


def profiles_handler(base_url, ctx):
    router = APIRouter(prefix=f'{base_url}/profiles')


    @router.post('', status_code=204)
    async def add_profile(user_profile: UserProfile):
        try:
            return await ctx.profile_service.add(user_profile)
        except EntityAlreadyExistsException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Already exist a profile with {user_profile.email} email. Cause: {e}"
            )


    @router.put('/{email}', status_code=200)
    async def update_profile(email: str, user_profile: UserProfile):
        return await ctx.profile_service.update(user_profile)


    @router.get("/{email}", status_code=200)
    async def get_profile(email: str):
        profile = await ctx.profile_service.find(email)

        if profile == None:
            raise HTTPException(status_code=404, detail=f'Not found {email} profile')
        else:
            return profile.model_dump(exclude_none=True)

    @router.get('', status_code=200)
    async def get_all_profiles():
        return await ctx.profile_service.all()


    @router.delete('/{email}', status_code=202)
    async def delete_profile(email: str):
        await ctx.profile_service.delete(email)
        return Response(status_code=202)


    return router