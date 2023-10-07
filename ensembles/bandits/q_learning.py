#
#
#
# -----------------------------------------------------------------------------------
# See: https://markelsanz14.medium.com/introducci%C3%B3n-al-aprendizaje-por-refuerzo-parte-2-q-learning-883cd42fb48e
# -----------------------------------------------------------------------------------
#
#
#
#
import numpy as np


class HorizontalPathEnvironment:
    """
         Scenario:

         (-5 points) step <- step <- step <- S -> step -> step -> step (+5 points)


         - A person could walk in two directions left or right
         - One step at a time.
         - into state could perform an step to left or right.
         - At the end of each direction reach a rewards negative at the left and positive at the right.
    """

    def __init__(self, left_reward=-5, right_reward=5, length=11, initial_state=5):
        steps = [0] * length
        steps[0] = left_reward
        steps[-1] = right_reward
        self.state_rewards = steps
        self.initial_state = initial_state
        self.current_state = self.initial_state

    @property
    def states_count(self):
        return len(self.state_rewards)

    def start_episode(self):
        self.current_state = self.initial_state

    def walk_on_step(self, direction):
        previous_state = self.current_state

        if direction > 0:
            self.current_state += 1
        else:
            self.current_state -= 1

        return self.state_rewards[self.current_state], previous_state

    @property
    def episode_end(self):
        rewards = self.state_rewards[self.current_state]
        return rewards == self.state_rewards[0] or rewards == self.state_rewards[-1]


class QLearningActionSelectionStrategy:
    def __init__(self, epsilon, discount, states_count):
        self.epsilon = epsilon
        self.discount = discount

        # (s,a) matrix. [left,right]
        self.q_values = [[0.0, 0.0] for _ in range(states_count)]

    def next_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, 2)  # Random action (left or right)
        else:
            return np.argmax(self.q_values[state])  # Greedy action for state

    def update_q_values(
            self,
            state,
            action,
            reward,
            next_state
    ):
        # Improve Q-values with Bellman Equation
        self.q_values[state][action] = reward + self.discount * max(self.q_values[next_state])

    def dec_epsilon(self, value):
        self.epsilon += value


def show_results(q_values):
    print(action_strategy.q_values)
    action_dict = {0: 'Left', 1: 'Right'}
    for state, Q_vals in enumerate(q_values):
        print(f'Best action on State {state}: Step to {action_dict[np.argmax(Q_vals)]}')


#
#
#
#
#
# -------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------
env = HorizontalPathEnvironment(
    left_reward=1,
    right_reward=5,
    length=7,
    initial_state=3
)

action_strategy = QLearningActionSelectionStrategy(
    epsilon=0.2,
    discount=0.9,  # Change to 1 to simplify Q-value results
    states_count=env.states_count
)

num_episodes = 300

for episode in range(num_episodes):
    env.start_episode()
    while not env.episode_end:  # Run until the end of the episode
        # Select action

        action = action_strategy.next_action(env.current_state)

        reward, previous_state = env.walk_on_step(direction=action)

        action_strategy.update_q_values(
            state=previous_state,
            action=action,
            reward=reward,
            next_state=env.current_state
        )
        # print(f'{previous_state} => {env.current_state}: {reward}')
        # print(action_strategy.q_values)

    if episode % 10 == 0:
        action_strategy.dec_epsilon(0.01)

show_results(action_strategy.q_values)
