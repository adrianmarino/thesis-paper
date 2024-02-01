import sys
from datetime import timedelta
from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.models import Variable

sys.path.append(f'{Variable.get("thesis.src_path")}')
sys.path.append(f'{Variable.get("thesis.src_path")}/../chat-bot-api')

from dag_task import cf_emb_update_task


with DAG(
    'CF-User-Item-Embeddings-Update',
    default_args       = {
        'owner'           : 'airflow',
        'depends_on_past' : False,
        'email'           : ['adrianmarino@gmail.com'],
        'email_on_failure': False,
        'email_on_retry'  : False,
        'retries'         : 3,
        'retry_delay'     : timedelta(minutes=120)
    },
    description       = """
        Generate and updated list of embeddings that represent users and items.
        Train a collaborative filtering model with a pre-train dataset and rec-chatbot
        user interactions. This training build embeddings for each user and item into
        rec-chatbot databases. Finally upsert embeddings into rec-chatbot chroma-db
        collections. These embeddings are used to build personalized recommendations.
    """,
    schedule_interval  = '*/5 * * * *',
    start_date         = days_ago(0),
    catchup            = False,
    max_active_runs    = 1,
    max_active_tasks   = 2,
    tags               = [
        'Deep-FM',
        'Rec-ChatBot API'
    ]
) as dag:
    cf_emb_update_task(dag)