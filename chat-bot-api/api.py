from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
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
    title='Recommendation Chatbot API',
    description='Provide personalized movie recommendations to users based on their profile, watched movies, and rating behavior.',
    version='0.9.0',
    contact={
        'name'  : 'Adrian Norberto Marino',
        'url'   : 'https://github.com/adrianmarino',
        'email' : 'adrianmarino@gmail.com',
    },
    swagger_ui_parameters={
        'deepLinking'           : True,
        'displayRequestDuration': True,
        'syntaxHighlight.theme' : 'monokai'
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


@app.on_event('startup')
async def startup_event():
    logger = logging.getLogger(__name__)
    logger.info("""


    ______            _____ _           _  ______       _
    | ___ \          /  __ \ |         | | | ___ \     | |
    | |_/ /___  ___  | /  \/ |__   __ _| |_| |_/ / ___ | |_
    |    // _ \/ __| | |   | '_ \ / _` | __| ___ \/ _ \| __|
    | |\ \  __/ (__  | \__/\ | | | (_| | |_| |_/ / (_) | |_
    \_| \_\___|\___|  \____/_| |_|\__,_|\__\____/ \___/ \__|


    --------------------------------------
    See docs:
     - Redoc..: http://0.0.0.0:8080/redoc
     - Swagger: http://0.0.0.0:8080/docs
    --------------------------------------
""")