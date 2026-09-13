
from .action_based import ActionBased


class SARSA(ActionBased):

    def __init__(
        self,
        actions=None,
        gamma=0.99,
        learning_rate=0.1
    ):
        self.actions = actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.q_table = {}

    def action_value(self, state, action):
        return self.q_table.get(
            (state, action),
            0.0
        )

    def td_target(
        self,
        reward,
        next_state,
        next_action,
        done=False
    ):
        if done:
            return reward

        if next_action is None:
            raise ValueError(
                "SARSA requires next_action"
            )

        return reward + self.gamma * self.action_value(
            next_state,
            next_action
        )

    def td_error(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        done=False
    ):
        value = self.action_value(
            state,
            action
        )

        target = self.td_target(
            reward,
            next_state,
            next_action,
            done
        )

        return target - value

    def update(self, experience):

        errors = []

        for transition in experience:

            state = transition.state
            action = transition.action
            reward = transition.reward
            next_state = transition.next_state
            next_action = transition.next_action
            done = transition.done

            error = self.td_error(
                state,
                action,
                reward,
                next_state,
                next_action,
                done
            )

            key = (state, action)

            self.q_table[key] = (
                self.action_value(
                    state,
                    action
                )
                + self.learning_rate * error
            )

            errors.append(error)

        return errors
