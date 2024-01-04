from fastapi import HTTPException, APIRouter, Response, Request
from models import UserMessage


def chats_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommendations')

  @router.post('')
  async def chat(user_message: UserMessage, request: Request):
    return await ctx.chat_bot_service.send(
      user_message,
      base_url=str(request.base_url)
    )

  return router