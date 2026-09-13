from .action_based import ActionBased


class MonteCarloControl(ActionBased):
    """First-visit tabular Monte-Carlo control over complete episodes."""

    def __init__(self, actions, gamma=0.99, learning_rate=None):
        self.actions = actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.q_table = {}
        self.counts = {}

    def action_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update(self, experience):
        transitions = list(experience)
        returns, running_return, seen = [0.0] * len(transitions), 0.0, set()
        for index in range(len(transitions) - 1, -1, -1):
            running_return = transitions[index].reward + self.gamma * running_return
            returns[index] = running_return
        for transition, episode_return in zip(transitions, returns):
            key = (transition.state, transition.action)
            if key in seen:
                continue
            seen.add(key)
            current = self.action_value(*key)
            if self.learning_rate is None:
                count = self.counts.get(key, 0) + 1
                self.counts[key] = count
                step_size = 1 / count
            else:
                step_size = self.learning_rate
            self.q_table[key] = current + step_size * (episode_return - current)
        return returns
