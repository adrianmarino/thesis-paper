import numpy as np

from .python_thesis_operator import python_thesis_operator


def python_callable(
    task_id,
    recsys_client_src_path,
    thesis_src_path,
    recsys_client_cfg_path,
    airflow_path,
    interactions_path,
    min_n_interactions,
    rating_scale
):
    import sys
    sys.path.append(recsys_client_src_path)
    sys.path.append(thesis_src_path)
    from recsys.domain_context import DomainContext
    import pandas as pd
    import service as srv
    import matplotlib.pyplot as plt

    domain = DomainContext(cfg_path=recsys_client_cfg_path)

    # --------------------------------------------------------------------------
    # Functions
    # --------------------------------------------------------------------------

    def save_interactions(df, name):
        df.to_json(
            f'{domain.cfg.temp_path}/{task_id}_{name}_interactions.json',
            orient='records'
        )

    def load_df(path_key):
        path_value = interactions_path if path_key == 'interactions_path' else path_key
        return pd.read_json(
            f'{domain.cfg.temp_path}/{path_value}',
            orient='records'
        )

    def train_predict(train_df, test_df, columns):
        model_loader = srv.GMFLoader(
            weights_path = domain.cfg.weights_path,
            metrics_path = domain.cfg.metrics_path,
            tmp_path     = domain.cfg.temp_path,
            user_seq_col = columns[0],
            item_seq_col = columns[1],
            rating_col   = columns[2],
            disable_plot = True
        )
        service = srv.ModulePredictionService(model_loader)
        service.predict(train_df, test_df)


    # --------------------------------------------------------------------------
    # Main Process
    # --------------------------------------------------------------------------

    interactions = load_df('interactions_path')

    # Build ratings matrix from user-item interactions..

    future_interactions, filtered_train_interactions = domain.interaction_inference_service.predict(
        train_interactions=interactions,
        columns=('user_seq', 'item_seq', 'rating'),
        train_predict_fn=lambda train, test, cols: train_predict(train, test, cols),
        min_n_interactions=min_n_interactions,
        rating_scale=rating_scale
    )

    future_interactions = future_interactions \
        .drop(columns=['rating']) \
        .rename(columns={'rating_prediction': 'rating'}) \
        .query('rating >= 0') # Saneamiento robusto para Dead ReLUs de PyTorch

    save_interactions(future_interactions, 'future')

    save_interactions(filtered_train_interactions, 'train')


def compute_gmf_rating_matrix_task(
        dag,
        task_id,
        interactions_path,
        min_n_interactions=20,
        rating_scale=np.arange(0, 6, 0.5)
):
    return python_thesis_operator(
        dag,
        task_id,
        python_callable,
        params={
            'interactions_path': interactions_path,
            'min_n_interactions': min_n_interactions,
            'rating_scale': rating_scale
        }
    )
