import numpy as np

from .python_thesis_operator import python_thesis_operator


def python_callable(**ctx):
    import sys
    sys.path.append(ctx['recsys.client.src_path'])
    sys.path.append(ctx['thesis.src_path'])
    from recsys.domain_context import DomainContext
    from recsys.dag.task       import ModuleComputeRatingMatrixTask

    task = ModuleComputeRatingMatrixTask(
        task_id      = ctx["task_id"],
        model_loader = srv.GMFLoader(
            weights_path = domain.cfg.weights_path,
            metrics_path = domain.cfg.metrics_path,
            tmp_path     = domain.cfg.temp_path
        ),
        domain       = DomainContext(cfg_path=ctx['recsys.client.cfg_path'])
    )

    task.perform(
        interactions_path  = ctx['interactions_path'],
        min_n_interactions = ctx['min_n_interactions'],
        rating_scale       = ctx['rating_scale']
    )


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
