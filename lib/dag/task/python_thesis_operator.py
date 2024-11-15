from airflow.operators.python import ExternalPythonOperator
from airflow.models import Variable
import logging

def python_thesis_operator(dag, task_id, python_callable, params = {}):
    op_kwargs = {
        'task_id'                 : task_id,
        'recsys.client.src_path'  : Variable.get('recsys.client.src_path'),
        'thesis.src_path'         : Variable.get('thesis.src_path'),
        'recsys.client.cfg_path'  : Variable.get('recsys.client.cfg_path'),
        'airflow_path'            : Variable.get('airflow_path')
    }
    op_kwargs.update(params)

    return ExternalPythonOperator(
        dag             = dag,
        python          = Variable.get('thesis.env_path'),
        task_id         = task_id,
        python_callable = python_callable,
        provide_context = True,
        do_xcom_push    = True,
        op_kwargs       = op_kwargs
    )