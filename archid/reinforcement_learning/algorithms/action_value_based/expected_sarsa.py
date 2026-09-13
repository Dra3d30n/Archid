from .action_based import ActionBased


class ExpectedSARSA(ActionBased):
    """Tabular on-policy TD control using the expected epsilon-greedy target."""

    def __init__(self, actions, gamma=0.99, learning_rate=0.1, epsilon=0.1):
        self.actions = actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = {}

    def action_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update(self, experience):
        errors = []
        for transition in experience:
            value = self.action_value(transition.state, transition.action)
            if transition.done:
                target = transition.reward
            else:
                actions = self.available_actions(transition.next_state)
                values = [self.action_value(transition.next_state, action) for action in actions]
                best = max(values)
                greedy_count = values.count(best)
                expected = sum(
                    (self.epsilon / len(actions) + ((1 - self.epsilon) / greedy_count if value == best else 0)) * value
                    for value in values
                )
                target = transition.reward + self.gamma * expected
            error = target - value
            self.q_table[transition.state, transition.action] = value + self.learning_rate * error
            errors.append(error)
        return errors
