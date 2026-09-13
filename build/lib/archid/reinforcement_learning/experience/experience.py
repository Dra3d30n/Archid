
class Transition:

    def __init__(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_action=None,
        log_prob=None
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.next_action = next_action
        self.log_prob = log_prob


class Experience:

    def __init__(self):
        self.transitions = []

    def add(self, transition):
        self.transitions.append(transition)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)

    def __iter__(self):
        return iter(self.transitions)

    def __getitem__(self, index):
        return self.transitions[index]
