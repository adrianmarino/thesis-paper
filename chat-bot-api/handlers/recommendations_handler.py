from fastapi import HTTPException, APIRouter, Response, Request
from models import UserMessage
import sys


def recommendations_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommendations')

  @router.post('')
  async def recommendations(
    request: Request,
    response: Response,
    user_message: UserMessage,
    plain: bool = False,
    metadata: bool = False,
    retry: int = 2,
    shuffle: bool = False
  ):
    while True:
      recommendations = await ctx.chat_bot_service.send(
        user_message,
        base_url=str(request.base_url),
        include_metadata=metadata,
        shuffle=shuffle
      )
      if not recommendations.empty or retry == 0:
        break
      retry -= 1

    if plain:
      return Response(content=recommendations.plain, media_type='text/plain')
    else:
      return [item.dict(exclude_none=True) for item in recommendations.items]


  return router