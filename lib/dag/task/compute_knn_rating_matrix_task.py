from .python_thesis_operator import python_thesis_operator
import numpy as np



def python_callable(**ctx):
    import sys
    sys.path.append(ctx['recsys.client.src_path'])
    sys.path.append(ctx['thesis.src_path'])
    from recsys.domain_context import DomainContext
    import util   as ut
    import pandas as pd
    import numpy  as np
    import service as sv
    import model as ml
    from scipy import sparse

    domain = DomainContext(cfg_path = ctx['recsys.client.cfg_path'])

    interactions = pd.read_json(
        f'{domain.cfg.temp_path}/{ctx["interactions_path"]}',
        orient='records'
    )


    def train_predict(train_df, test_df, columns, model):
        if 'knn_user_based' == model:
            predictor_name = 'knn_user_based'
            model_Type     = ml.KNNType.USER_BASED
        else:
            predictor_name = 'knn_item_based'
            model_Type     = ml.KNNType.ITEM_BASED

        predictor =  sv.KNNPredictionService(
            weights_path   = domain.cfg.weights_path,
            temp_path      = domain.cfg.temp_path,
            predictor_name = predictor_name,
            user_seq_col   = columns[0],
            item_seq_col   = columns[1],
            rating_col     = columns[2],
            model_Type     = model_Type
        )
        predictor.fit_predict(train_df, test_df)
        predictor.delete()


    # Build ratings matrix from user-item interactions..
    rating_matrix, _ = domain.rating_matrix_service.create(
        interactions,
        train_predict_fn   = lambda train, test, cols: train_predict(train, test, cols, ctx['model']),
        min_n_interactions = ctx['min_n_interactions'],
        rating_scale       = ctx['rating_scale']
    )

    sparse.save_npz(
        f'{domain.cfg.temp_path}/{ctx["task_id"]}.npz',
        rating_matrix
    )



def compute_knn_rating_matrix_task(
    dag,
    task_id,
    interactions_path,
    model              = 'knn_user_based',
    min_n_interactions = 20,
    rating_scale       = np.arange(0, 6, 0.5)
):
    return python_thesis_operator(
        dag,
        task_id,
        python_callable,
        params = {
            'interactions_path'  : interactions_path,
            'model'              : model,
            'min_n_interactions' : min_n_interactions,
            'rating_scale'       : rating_scale
        }
    )

