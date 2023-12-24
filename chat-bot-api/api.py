from fastapi import FastAPI, HTTPException, APIRouter
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