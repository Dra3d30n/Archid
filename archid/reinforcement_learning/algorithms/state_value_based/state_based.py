from ..algorithm import Algorithm


class StateBased(Algorithm):

    def state_value(self, state):
        raise NotImplementedError

    def td_target(
        self,
        reward,
        next_state,
        done
    ):
        if done:
            return reward

        return reward + self.gamma * self.state_value(
            next_state
        )

    def td_error(
        self,
        state,
        reward,
        next_state,
        done
    ):
        value = self.state_value(state)
        target = self.td_target(
            reward,
            next_state,
            done
        )
        return target - value


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

        error = self.td_error(
            experience.state,
            experience.reward,
            experience.next_state,
            experience.done
        )

        state = experience.state

        self.v_table[state] = (
            self.state_value(state)
            + self.learning_rate * error
        )

        return error