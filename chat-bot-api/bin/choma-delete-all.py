import os
import pytorch_common.util as pu
import chromadb
import logging

pu.LoggerBuilder().on_console().build()

client = chromadb.HttpClient(
    host=os.environ['CHROMA_HOST'],
    port=os.environ['CHROMA_PORT']
)

logging.info(f'Start: Delete all chroma db collections...')
deleted = 0

for collection in client.list_collections():
    try:
        client.delete_collection(collection.name)
        logging.info(f'==> "{collection.name}" collection deleted...')
        deleted += 1
    except Exception as e:
        logging.error(f"Can't remove {collection.name} collection. Detail: {e}")

logging.info(f'Finish: {deleted} collections deleted')