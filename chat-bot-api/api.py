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
    items_handler,
    recommenders_handler
)


BASE_URL = '/api/v1'

# Setup loggers
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

app = FastAPI(
    title="Rec ChatBot API",
    description='Allow recommends movies to users base on your profile and seen movies.',
    version="0.5.0",
    contact={
        "name": "Adria Norberto Marino",
        "url": "https://github.com/adrianmarino",
        "email": "adrianmarino@gmail.com",
    }
)

ctx = AppContext()

app.include_router(profiles_handler(BASE_URL, ctx))
app.include_router(recommendations_handler(BASE_URL, ctx))
app.include_router(chat_histories_handler(BASE_URL, ctx))
app.include_router(interactions_handler(BASE_URL, ctx))
app.include_router(recommenders_handler(BASE_URL, ctx))


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



@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger(__name__)
    logger.info("""

    _____              _____ _                ____        _
    |  __ \            / ____| |              |  _ \      | |
    | |__) |___  ___  | |    | |__   __ _ _ __| |_) | ___ | |_
    |  _  // _ \/ __| | |    | '_ \ / _` | '__|  _ < / _ \| __|
    | | \ \  __/ (__  | |____| | | | (_| | |  | |_) | (_) | |_
    |_|  \_\___|\___|  \_____|_| |_|\__,_|_|  |____/ \___/ \__|

    ------------------------------------------
    See docs:
     - Redoc......: http://0.0.0.0:8080/redoc
     - Swagger Doc: http://0.0.0.0:8080/docs
    ------------------------------------------
""")