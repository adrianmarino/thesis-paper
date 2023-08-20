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
    'Thesis-Recommenders-Upgrade',
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
        'thesis',
        'distance-matrix-computing',
        'knn_user_based',
        'knn_item_based'
    ]
) as dag:

    fetch = ts.fetch_interactions_task(dag, task_id = 'fetch_interactions')

    check_count = ts.check_interactions_change_task(dag, interactions_path = 'fetch_interactions.json')

    check_branch = ts.branch_task(
        dag,
        true_task_id = [
            'compute_svd_rating_matrix',
            'compute_nmf_rating_matrix',
            'compute_knn_user_based_rating_matrix',
            'compute_knn_item_based_rating_matrix'
        ]
    )

    mark = ts.mark_n_processed_interactions_task(dag, interactions_path = 'fetch_interactions.json')

    knn_user_based_rating_matrix = tss.compute_knn_rating_matrix_task(
        dag,
        task_id           = 'compute_knn_user_based_rating_matrix',
        interactions_path = 'fetch_interactions.json',
        model             = 'knn_user_based',
        min_n_interactions = 20,
        rating_scale       = np.arange(0, 6, 0.5)
    )

    knn_item_based_rating_matrix = tss.compute_knn_rating_matrix_task(
        dag,
        task_id           = 'compute_knn_item_based_rating_matrix',
        interactions_path = 'fetch_interactions.json',
        model             = 'knn_item_based',
        min_n_interactions = 20,
        rating_scale       = np.arange(0, 6, 0.5)
    )

    svd_rating_matrix = ts.compute_surprise_rating_matrix_task(
        dag,
        task_id           = 'compute_svd_rating_matrix',
        interactions_path = 'fetch_interactions.json',
        model             = 'SVD',
        min_n_interactions = 20,
        rating_scale       = np.arange(0, 6, 0.5)
    )

    nmf_rating_matrix = ts.compute_surprise_rating_matrix_task(
        dag,
        task_id           = 'compute_nmf_rating_matrix',
        interactions_path = 'fetch_interactions.json',
        model             = 'NMF',
        min_n_interactions = 20,
        rating_scale       = np.arange(0, 6, 0.5)
    )

    knn_user_based_sim = ts.compute_similarities_task(
        dag,
        task_id                  = 'compute_knn_user_based_similarities',
        future_interactions_path = 'compute_knn_user_based_rating_matrix_future_interactions.json',
        train_interactions_path  = 'compute_knn_user_based_rating_matrix_train_interactions.json'
    )

    knn_item_based_sim = ts.compute_similarities_task(
        dag,
        task_id                  = 'compute_knn_item_based_similarities',
        future_interactions_path = 'compute_knn_item_based_rating_matrix_future_interactions.json',
        train_interactions_path  = 'compute_knn_item_based_rating_matrix_train_interactions.json'
    )

    svd_sim = ts.compute_similarities_task(
        dag,
        task_id                  = 'compute_svd_similarities',
        future_interactions_path = 'compute_svd_rating_matrix_future_interactions.json',
        train_interactions_path  = 'compute_svd_rating_matrix_train_interactions.json'
    )

    nmf_sim = ts.compute_similarities_task(
        dag,
        task_id                  = 'compute_nmf_similarities',
        future_interactions_path = 'compute_nmf_rating_matrix_future_interactions.json',
        train_interactions_path  = 'compute_nmf_rating_matrix_train_interactions.json'
    )

    upgrade_svd_rec = ts.update_recommender_task(
        dag,
        task_id                 = 'update_svd_recommender',
        recommender_name        = 'SVD',
        interactions_path       = 'compute_svd_rating_matrix_train_interactions.json',
        user_similarities_path  = 'compute_svd_similarities_user_similarities.json',
        item_similarities_path  = 'compute_svd_similarities_item_similarities.json',
        n_most_similars_users   = 500,
        n_most_similars_items   = 10
    )

    upgrade_nmf_rec = ts.update_recommender_task(
        dag,
        task_id                 = 'update_nmf_recommender',
        recommender_name        = 'NMF',
        interactions_path       = 'compute_nmf_rating_matrix_train_interactions.json',
        user_similarities_path  = 'compute_nmf_similarities_user_similarities.json',
        item_similarities_path  = 'compute_nmf_similarities_item_similarities.json',
        n_most_similars_users   = 500,
        n_most_similars_items   = 10
    )


    upgrade_knn_user_based_rec = ts.update_recommender_task(
        dag,
        task_id                 = 'update_knn_user_based_recommender',
        recommender_name        = 'knn_user_based',
        interactions_path       = 'compute_knn_user_based_rating_matrix_train_interactions.json',
        user_similarities_path  = 'compute_knn_user_based_similarities_user_similarities.json',
        item_similarities_path  = 'compute_knn_user_based_similarities_item_similarities.json',
        n_most_similars_users   = 500,
        n_most_similars_items   = 10
    )

    upgrade_knn_item_based_rec = ts.update_recommender_task(
        dag,
        task_id                 = 'update_knn_item_based_recommender',
        recommender_name        = 'knn_item_based',
        interactions_path       = 'compute_knn_item_based_rating_matrix_train_interactions.json',
        user_similarities_path  = 'compute_knn_item_based_similarities_user_similarities.json',
        item_similarities_path  = 'compute_knn_item_based_similarities_item_similarities.json',
        n_most_similars_users   = 500,
        n_most_similars_items   = 10
    )

    fetch >> check_count >> check_branch


    check_branch >> svd_rating_matrix >> svd_sim >> upgrade_svd_rec
    check_branch >> nmf_rating_matrix >> nmf_sim >> upgrade_nmf_rec

    check_branch >> knn_user_based_rating_matrix >> knn_user_based_sim >> upgrade_knn_user_based_rec
    check_branch >> knn_item_based_rating_matrix >> knn_item_based_sim >> upgrade_knn_item_based_rec

    [upgrade_svd_rec, upgrade_nmf_rec, upgrade_knn_user_based_rec, upgrade_knn_item_based_rec] >> mark
