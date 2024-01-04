from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import logging
from app_context import AppContext
from handlers import (
    profiles_handler,
    recommendations_handler,
    chat_histories_handler,
    interactions_handler,
    items_handler
)


BASE_URL = '/api/v1'

# Setup loggers
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

app = FastAPI()
ctx = AppContext()

profiles_router = profiles_handler(BASE_URL, ctx)
app.include_router(profiles_router)


recommendations_router = recommendations_handler(BASE_URL, ctx)
app.include_router(recommendations_router)


chat_histories_router = chat_histories_handler(BASE_URL, ctx)
app.include_router(chat_histories_router)


interactions_router = interactions_handler(BASE_URL, ctx)
app.include_router(interactions_router)


items_router = items_handler(BASE_URL, ctx)
app.include_router(items_router)


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