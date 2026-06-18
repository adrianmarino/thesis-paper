from datetime import timedelta
from airflow import DAG
from airflow.models import Variable
import pendulum
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from dag_utils import BashTaskBuilder


with DAG(
        'all-mpnet-base-v2-bert-item-distance-matrix',
        default_args      = {
                'owner'           : 'adrian',
                'depends_on_past' : False,
                'retries'         : 10,
                'retry_delay'     : timedelta(minutes=3),
        },
        description       = 'all-mpnet-base-v2-bert-item-distance-matrix',
        schedule = '*/10 * * * *',
        start_date        = pendulum.today('UTC'),
        catchup           = False,
        tags              = ['rec-sys'],
        max_active_runs   = 1
) as dag:
        # Create all tasks...
        job_task = BashTaskBuilder('all-mpnet-base-v2-bert-item-distance-matrix-task') \
                .script('python bin/all_mpnet_base_v2_bert_item_distance_matrix_job.py') \
                .build()

        # Workflow...
        job_task
