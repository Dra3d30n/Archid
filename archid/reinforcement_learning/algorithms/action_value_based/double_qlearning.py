import random

from .action_based import ActionBased


class DoubleQLearning(ActionBased):
    """Tabular Double Q-learning, reducing maximization bias with two tables."""

    def __init__(self, actions, gamma=0.99, learning_rate=0.1):
        self.actions = actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.q_table_a = {}
        self.q_table_b = {}

    def action_value(self, state, action):
        return self.q_table_a.get((state, action), 0.0) + self.q_table_b.get((state, action), 0.0)

    def update(self, experience):
        errors = []
        for transition in experience:
            first, second = (self.q_table_a, self.q_table_b) if random.random() < 0.5 else (self.q_table_b, self.q_table_a)
            key = (transition.state, transition.action)
            value = first.get(key, 0.0)
            if transition.done:
                target = transition.reward
            else:
                actions = self.available_actions(transition.next_state)
                best_action = max(actions, key=lambda action: first.get((transition.next_state, action), 0.0))
                target = transition.reward + self.gamma * second.get((transition.next_state, best_action), 0.0)
            error = target - value
            first[key] = value + self.learning_rate * error
            errors.append(error)
        return errors
