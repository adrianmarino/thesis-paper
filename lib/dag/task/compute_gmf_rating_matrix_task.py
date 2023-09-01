import numpy as np

from .python_thesis_operator import python_thesis_operator


def python_callable(**ctx):
    import sys
    sys.path.append(ctx['recsys.client.src_path'])
    sys.path.append(ctx['thesis.src_path'])
    from recsys.domain_context import DomainContext
    import pandas as pd
    import service as srv
    import logging

    domain = DomainContext(cfg_path=ctx['recsys.client.cfg_path'])

    # --------------------------------------------------------------------------
    # Functions
    # --------------------------------------------------------------------------

    def save_interactions(df, name):
        df.to_json(
            f'{domain.cfg.temp_path}/{ctx["task_id"]}_{name}_interactions.json',
            orient='records'
        )

    def load_df(path):
        return pd.read_json(
            f'{domain.cfg.temp_path}/{ctx[path]}',
            orient='records'
        )

    def train_predict(train_df, test_df, columns):
        model_loader = srv.GMFLoader(
            weights_path = domain.cfg.weights_path,
            metrics_path = domain.cfg.metrics_path,
            tmp_path     = domain.cfg.temp_path,
            user_seq_col = columns[0],
            item_seq_col = columns[1],
            rating_col   = columns[2]
        )
        service = srv.ModulePredictionService(model_loader)

        service.predict(train_df, test_df)


    # --------------------------------------------------------------------------
    # Main Process
    # --------------------------------------------------------------------------

    interactions = load_df('interactions_path')

    # Build ratings matrix from user-item interactions..

    logging.info(f'Columns: {interactions.columns}')

    future_interactions, filtered_train_interactions = domain.interaction_inference_service.predict(
        train_interactions=interactions,
        columns=('user_seq', 'item_seq', 'rating'),
        train_predict_fn=lambda train, test, cols: train_predict(train, test, cols),
        min_n_interactions=ctx['min_n_interactions'],
        rating_scale=ctx['rating_scale']
    )

    future_interactions = future_interactions \
        .drop(columns=['rating']) \
        .rename(columns={'rating_prediction': 'rating'}) \
        .query('rating > 0')

    logging.info(f'Columns: {future_interactions.columns}')

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
