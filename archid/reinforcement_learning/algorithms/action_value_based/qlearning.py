
from .action_based import ActionBased


class QLearning(ActionBased):

    def __init__(
        self,
        actions,
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
        done=False
    ):
        if done:
            return reward

        next_actions = self.actions(next_state)

        if not next_actions:
            return reward

        return reward + self.gamma * max(
            self.action_value(
                next_state,
                action
            )
            for action in next_actions
        )

    def td_error(
        self,
        state,
        action,
        reward,
        next_state,
        done=False
    ):
        value = self.action_value(
            state,
            action
        )

        target = self.td_target(
            reward,
            next_state,
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
            done = transition.done

            error = self.td_error(
                state,
                action,
                reward,
                next_state,
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