from .action_based import ActionBased


class DQN(ActionBased):

    def __init__(
        self,
        critic,
        critic_optimizer,
        actions,
        gamma=0.99
    ):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.actions = actions
        self.gamma = gamma

    def action_value(self, state, action):
        return self.critic.forward(
            state,
            action
        )

    def td_target(
        self,
        reward,
        next_state,
        done=False
    ):
        if done:
            return reward

        next_values = [
            self.action_value(
                next_state,
                action
            )
            for action in self.actions
        ]

        # ``next_values`` are Euclid scalar Tensors, which are not orderable
        # by Python's built-in ``max``.  Select using detached numeric data;
        # the bootstrap target must not backpropagate through this branch.
        best_next_value = max(
            next_values,
            key=lambda value: float(value.data.reshape(-1)[0])
        )

        # A Q-learning target is a constant for this update. Keeping the
        # Euclid graph here causes gradients to flow through both sides of the
        # Bellman error and makes the online critic unstable.
        return reward + self.gamma * float(best_next_value.data.reshape(-1)[0])

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

            loss = error ** 2

            self.critic_optimizer.zero_grad()

            loss.backward()

            self.critic_optimizer.step()

        return loss
