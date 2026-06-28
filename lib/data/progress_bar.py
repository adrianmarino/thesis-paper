import os

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

if run_from_ipython():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def progress_bar(count, title='Processing'):
    is_airflow = 'AIRFLOW_CTX_TASK_ID' in os.environ
    return tqdm(total=count, desc=title, disable=is_airflow)
