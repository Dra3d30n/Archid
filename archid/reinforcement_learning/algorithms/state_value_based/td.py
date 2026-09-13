
from .state_based import StateBased


class TD(StateBased):

    def __init__(
        self,
        learning_rate=0.1,
        gamma=0.99
    ):
        super().__init__(
            gamma=gamma
        )

        self.learning_rate = learning_rate
        self.v_table = {}

    def state_value(self, state):
        return self.v_table.get(
            state,
            0.0
        )

    def update(self, experience):

        errors = []

        for transition in experience:

            state = transition.state
            reward = transition.reward
            next_state = transition.next_state
            done = transition.done

            error = self.td_error(
                state,
                reward,
                next_state,
                done
            )

            self.v_table[state] = (
                self.state_value(state)
                + self.learning_rate * error
            )

            errors.append(error)

        return errors
