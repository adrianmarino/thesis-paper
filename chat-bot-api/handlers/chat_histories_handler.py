from fastapi import HTTPException, APIRouter


def chat_histories_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/histories')

  @router.get('/{email}')
  async def get_history(email: str):
    history = await ctx.history_service.find(email)

    if history == None:
        raise HTTPException(status_code=404, detail=f'Not found {email} history')
    else:
        return history

  @router.delete('/{email}', status_code=204)
  async def delete_history(email: str):
      return await ctx.history_service.delete(email)


  return router