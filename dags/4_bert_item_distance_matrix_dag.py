from datetime import timedelta
from airflow import DAG
from airflow.models import Variable
import pendulum
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from dag_utils import BashTaskBuilder


with DAG(
        'multi-qa-mpnet-base-dot-v1-bert-item-distance-matrix',
        default_args      = {
                'owner'           : 'adrian',
                'depends_on_past' : False,
                'retries'         : 10,
                'retry_delay'     : timedelta(minutes=3),
        },
        description       = 'multi-qa-mpnet-base-dot-v1-bert-item-distance-matrix',
        schedule = '*/10 * * * *',
        start_date        = pendulum.today('UTC'),
        catchup           = False,
        tags              = ['rec-sys'],
        max_active_runs   = 1
) as dag:
        # Create all tasks...
        job_task = BashTaskBuilder('multi-qa-mpnet-base-dot-v1-bert-item-distance-matrix-task') \
                .script('python bin/multi_qa_mpnet_base_dot_v1_bert_item_distance_matrix_job.py') \
                .build()

        # Workflow...
        job_task
