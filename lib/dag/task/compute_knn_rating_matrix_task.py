import numpy as np

from .python_thesis_operator import python_thesis_operator


def python_callable(**ctx):
    import sys
    sys.path.append(ctx['recsys.client.src_path'])
    sys.path.append(ctx['thesis.src_path'])
    from recsys.domain_context import DomainContext
    import pandas as pd
    import service as sv
    import model as ml

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
        return pd.read_json(f'{domain.cfg.temp_path}/{ctx[path]}', orient='records')

    def train_predict(train_df, test_df, columns, model):
        model_type = ml.KNNType.USER_BASED if 'knn_user_based' == model else ml.KNNType.ITEM_BASED

        knn_loader = sv.KNNLoader(
            weights_path=domain.cfg.weights_path,
            temp_path=domain.cfg.temp_path,
            predictor_name=model,
            user_seq_col=columns[0],
            item_seq_col=columns[1],
            rating_col=columns[2],
            update_period_in_minutes=180,  # 3 hours,
            model_type=model_type
        )

        sv.KNNPredictionService(
            knn_loader,
            user_seq_col=columns[0],
            item_seq_col=columns[1],
            rating_col=columns[2]
        )(
            train_df,
            test_df,
            n_neighbors=50
        )

    # --------------------------------------------------------------------------
    # Main Process
    # --------------------------------------------------------------------------

    interactions = load_df('interactions_path')

    # Build ratings matrix from user-item interactions..
    future_interactions, filtered_train_interactions = domain.interaction_inference_service.predict(
        train_interactions=interactions,
        columns=('user_seq', 'item_seq', 'rating'),
        train_predict_fn=lambda train, test, cols: train_predict(train, test, cols, ctx['model']),
        min_n_interactions=ctx['min_n_interactions'],
        rating_scale=ctx['rating_scale']
    )

    del interactions

    future_interactions = future_interactions.rename(columns={'rating_prediction': 'rating'})

    future_interactions = future_interactions[future_interactions['rating'] >= 0]

    save_interactions(future_interactions, 'future')
    del future_interactions

    save_interactions(filtered_train_interactions, 'train')
    del filtered_train_interactions


def compute_knn_rating_matrix_task(
        dag,
        task_id,
        interactions_path,
        model='knn_user_based',
        min_n_interactions=20,
        rating_scale=np.arange(0, 6, 0.5)
):
    return python_thesis_operator(
        dag,
        task_id,
        python_callable,
        params={
            'interactions_path': interactions_path,
            'model': model,
            'min_n_interactions': min_n_interactions,
            'rating_scale': rating_scale
        }
    )
