import requests
from fastapi import Depends
from fastapi_health import health
import logging
import os

MONGO_URL = f'http://{os.environ["MONGODB_HOST"]}:{os.environ["MONGODB_PORT"]}'



def health_handler(app, base_url, ctx):
    def is_ollama_active():
        non_healthy = { 'ollama_api': False }
        try:
            response = requests.get('http://nonosoft.ddns.net:7070/ollama/')
            if  response.status_code != 200:
                return non_healthy
            else:
                return { 'ollama_api': response.content.decode("utf-8") == 'Ollama is running' }
        except:
            return non_healthy


    def is_airflow_active():
        non_healthy = { 'airflow': {'metadatabase': False, 'scheduler': False} }
        try:
            response = requests.get('http://nonosoft.ddns.net:8686/health')
            if  response.status_code != 200:
                return non_healthy

            dto = response.json()
            return {
                'airflow': {
                    'metadatabase': dto['metadatabase']['status'] == "healthy",
                    'scheduler': dto['scheduler']['status'] ==  "healthy",
                }
            }
        except:
            return non_healthy


    def is_chomadb_database_active():
        def  is_chroma_active():
            try:
                return ctx.items_content_emb_repository.count() > 0
            except:
                return False
        return { 'choma_database': is_chroma_active() }


    def is_mongodb_database_active():
        try:
            return { 'mongo_database':  requests.get(MONGO_URL).status_code == 200}
        except:
            return { 'mongo_database': False }


    def api_active(
        ollama: bool = Depends(is_ollama_active),
        airflow: bool = Depends(is_airflow_active),
        chroma: bool = Depends(is_chomadb_database_active),
        mongo: bool = Depends(is_mongodb_database_active),
    ):
        return {
            'chatbot_api': ollama['ollama_api'] and airflow['airflow'] and chroma['choma_database'] and mongo['mongo_database']
        }


    app.add_api_route(
        f'{base_url}/health',
        health([
            api_active,
            is_ollama_active,
            is_airflow_active,
            is_mongodb_database_active,
            is_chomadb_database_active
        ])
    )