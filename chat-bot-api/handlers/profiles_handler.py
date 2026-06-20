from models import UserProfile
from fastapi import HTTPException, APIRouter, Response
from repository.mongo import EntityAlreadyExistsException
import logging


def profiles_handler(base_url, ctx):
    router = APIRouter(prefix=f'{base_url}/profiles', tags=["Profiles"])


    @router.post('', status_code=204, summary="Create a new User Profile")
    async def add_profile(user_profile: UserProfile):
        """
        Creates a new user profile.
        This is a critical step to solve the **Cold-Start** problem in recommendation systems.
        By providing initial preferences (e.g., preferred genres, release years), the engine can recommend items even if the user hasn't rated any movie yet.
        """
        try:
            return await ctx.profile_service.add(user_profile)
        except EntityAlreadyExistsException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Already exist a profile with {user_profile.email} email. Cause: {e}"
            )


    @router.put('/{email}', status_code=200, summary="Update an existing User Profile")
    async def update_profile(email: str, user_profile: UserProfile):
        """
        Updates an existing user profile's preferences or metadata based on their email.
        """
        return await ctx.profile_service.update(user_profile)


    @router.get("/{email}", status_code=200, summary="Get a User Profile by Email")
    async def get_profile(email: str):
        """
        Retrieves a user's profile details.
        """
        profile = await ctx.profile_service.find(email)

        if profile == None:
            raise HTTPException(status_code=404, detail=f'Not found {email} profile')
        else:
            return profile.model_dump(exclude_none=True)

    @router.get('', status_code=200, summary="Get all User Profiles")
    async def get_all_profiles():
        """
        Retrieves a list of all registered user profiles.
        """
        return await ctx.profile_service.all()


    @router.delete('/{email}', status_code=202, summary="Delete a User Profile")
    async def delete_profile(email: str):
        """
        Removes a user profile from the database.
        """
        await ctx.profile_service.delete(email)
        return Response(status_code=202)


    return router
