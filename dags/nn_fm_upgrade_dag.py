import sys
from datetime import timedelta
from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.models import Variable

sys.path.append(Variable.get('thesis.src_path'))
sys.path.append(Variable.get('recsys.client.src_path'))

import recsys.dag.task as ts
import dag.task as tss
import numpy as np


with DAG(
    'NN-FM-Recommender-Upgrade',
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
        This DAG perform next steps: Fetch all user interactions from rec-sys API,
        filter interactions for users with more that 20 interactions.
        Then for each model: train it using user interactions, compute rating matrix
        for all interactions real and predicted. Use rating matrix to compute user-user
        and item-item similarity matrix and upsert matrix to rec-sys API.
        Finally, compute user-user and item-item similarity matrix using mean from
        all models similarity matrix and upsert to rec-sys API.
    """,
    schedule_interval  = '*/10 * * * *',
    start_date         = days_ago(0),
    catchup            = False,
    max_active_runs    = 1,
    max_active_tasks   = 2,
    tags               = [
        'NNFM',
        'rec-sys'
    ]
) as dag:

    fetch = ts.fetch_interactions_task(dag, task_id = 'fetch_interactions')

    nn_fm_rating_matrix = tss.compute_nn_fm_rating_matrix_task(
        dag,
        task_id           = 'compute_nn_fm_rating_matrix',
        interactions_path = 'fetch_interactions.json',
        min_n_interactions = 20,
        rating_scale       = np.arange(0, 6, 0.5)
    )

    nn_fm_sim = ts.compute_similarities_task(
        dag,
        task_id                  = 'compute_nn_fm_similarities',
        future_interactions_path = 'compute_nn_fm_rating_matrix_future_interactions.json',
        train_interactions_path  = 'compute_nn_fm_rating_matrix_train_interactions.json'
    )

    upgrade_nn_fm_rec = ts.update_recommender_task(
        dag,
        task_id                 = 'update_nn_fm_recommender',
        recommender_name        = 'NNFM',
        interactions_path       = 'compute_nn_fm_rating_matrix_train_interactions.json',
        user_similarities_path  = 'compute_nn_fm_similarities_user_similarities.json',
        item_similarities_path  = 'compute_nn_fm_similarities_item_similarities.json',
        n_most_similars_users   = 500,
        n_most_similars_items   = 10
    )

    fetch >> nn_fm_rating_matrix >> nn_fm_sim >> upgrade_nn_fm_rec
