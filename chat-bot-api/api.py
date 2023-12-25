from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app_context import AppContext
from handlers import profiles_handler, chats_handler, chat_histories_handler

BASE_URL = '/api/v1'

app = FastAPI()
ctx = AppContext()

profiles_router = profiles_handler(BASE_URL, ctx)
app.include_router(profiles_router)


chats_router = chats_handler(BASE_URL, ctx)
app.include_router(chats_router)


chat_histories_router = chat_histories_handler(BASE_URL, ctx)
app.include_router(chat_histories_router)





@app.exception_handler(500)
async def internal_exception_handler(request: Request, e: Exception):
  return JSONResponse(
    status_code = 500, 
    content     = jsonable_encoder(
        {
            'code': 500, 
            'msg': f'Internal Server Error. Cause: {e}',
        }
    )
 )