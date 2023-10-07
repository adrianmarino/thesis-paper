from .experiment_tracker import ExperimentTracker


def experiment(group, selection_strategy, params):
    tracker = ExperimentTracker(group, params.n_iterations)

    for iteration in range(params.n_iterations + 1):
        machine = group.select_machine(selection_strategy)

        machine.pull_arm()

        machine.gain_prob = machine.gain / float(machine.attempts)

        if iteration % 10 == 0:
            selection_strategy.decrease_exploration(params.epsilon_delta)

        tracker.save_logs(iteration)

    return tracker
