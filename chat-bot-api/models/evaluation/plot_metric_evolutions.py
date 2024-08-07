import data.plot as dp


def plot_mean_ndcg_evolutions(
    llama2_sessions,
    llama3_sessions,
    moving_avg_window = 5,
    min_sessions_by_step = 10
):
    dp.plot_lines(
        lines             = {
            'Llama3': list(llama3_sessions.steps_by_index.mean_ndcg(min_sessions_by_step).values()),
            'Llama2': list(llama2_sessions.steps_by_index.mean_ndcg(min_sessions_by_step).values())
        },
        hue               = 'LLM',
        xlabel            = 'User Session Step',
        ylabel            = 'Mean NDGC',
        title             = 'Mean NDGC by Users Sessions Step',
        moving_avg_window = moving_avg_window,
        fill              = True,
    )

def plot_mean_avg_precision_evolutions(
    llama2_sessions,
    llama3_sessions,
    moving_avg_window = 5,
    min_sessions_by_step = 10
):
    dp.plot_lines(
        lines             = {
            'Llama3': list(llama3_sessions.steps_by_index.mean_average_precision(min_sessions_by_step).values()),
            'Llama2': list(llama2_sessions.steps_by_index.mean_average_precision(min_sessions_by_step).values()),
        },
        hue               = 'LLM',
        xlabel            = 'User Session Step',
        ylabel            = 'Mean Average Precision',
        title             = 'Mean Average Precision by Users Sessions Step',
        moving_avg_window = moving_avg_window,
        fill              = True,
    )

def plot_mean_reciprocal_rank_evolutions(
    llama2_sessions,
    llama3_sessions,
    moving_avg_window = 5,
    min_sessions_by_step = 10
):
    dp.plot_lines(
        lines             = {
            'Llama3': list(llama3_sessions.steps_by_index.mean_reciprocal_rank(min_sessions_by_step).values()),
            'Llama2': list(llama2_sessions.steps_by_index.mean_reciprocal_rank(min_sessions_by_step).values())
        },
        hue               = 'LLM',
        xlabel            = 'User Session Step',
        ylabel            = 'Mean Reciprocal Rank',
        title             = 'Mean Reciprocal Rank by Users Sessions Step',
        moving_avg_window = moving_avg_window,
        fill              = True,
    )

def plot_mean_recall_evolutions(
    llama2_sessions,
    llama3_sessions,
    moving_avg_window = 5,
    min_sessions_by_step = 10
):
    dp.plot_lines(
        lines             = {
            'Llama3': list(llama3_sessions.steps_by_index.mean_recall(min_sessions_by_step).values()),
            'Llama2': list(llama2_sessions.steps_by_index.mean_recall(min_sessions_by_step).values())
        },
        hue               = 'LLM',
        xlabel            = 'User Session Step',
        ylabel            = 'Mean Recall',
        title             = 'Mean Recall by Users Sessions Step',
        moving_avg_window = moving_avg_window,
        fill              = True,
    )