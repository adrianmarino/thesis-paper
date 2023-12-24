from models import UserProfile
from fastapi import HTTPException, APIRouter


def profiles_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/profiles')

  @router.post('', status_code=204)
  async def add_profile(userProfile: UserProfile):
      try:
          await ctx.profiles_repository.add_one(userProfile)
          return await ctx.profiles_repository.find_by_id(userProfile.email)
      except Exception as e:
          raise HTTPException(status_code=400, detail="user email already registered")

  @router.put('/{email}', status_code=200)
  async def update_profile(email: str, userProfile: UserProfile):
      await ctx.profiles_repository.update(userProfile)
      return await ctx.profiles_repository.find_by_id(email)


  @router.get("/{email}", status_code=200)
  async def get_profile(email: str):
    profile = await ctx.profiles_repository.find_by_id(email)

    if profile == None:
        raise HTTPException(status_code=404, detail=f'Not found {email} profile')
    else:
        return profile

  @router.get('', status_code=200)
  async def get_all_profile():
      return await ctx.profiles_repository.find_all()


  return router