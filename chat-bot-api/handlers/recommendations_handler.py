from fastapi import HTTPException, APIRouter, Response, Request
from models import UserMessage


def recommendations_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommendations')

  @router.post('')
  async def recommendations(user_message: UserMessage, request: Request, metadata: bool = False):
    recommendations = await ctx.chat_bot_service.send(
      user_message,
      base_url=str(request.base_url),
      include_metadata=metadata
    )

    return [r.dict(exclude_none=True) for r in recommendations]

  return router