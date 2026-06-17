from airflow.providers.standard.operators.python import ExternalPythonOperator
from airflow.models import Variable
import logging

def python_thesis_operator(dag, task_id, python_callable, params = {}):
    op_kwargs = {
        'task_id'                 : task_id,
        'recsys_client_src_path'  : Variable.get('recsys.client.src_path'),
        'thesis_src_path'         : Variable.get('thesis.src_path'),
        'recsys_client_cfg_path'  : Variable.get('recsys.client.cfg_path'),
        'airflow_path'            : Variable.get('airflow_path')
    }
    op_kwargs.update(params)

    return ExternalPythonOperator(
        dag             = dag,
        python          = Variable.get('thesis.env_path'),
        task_id         = task_id,
        python_callable = python_callable,

        do_xcom_push    = True,
        op_kwargs       = op_kwargs
    )
