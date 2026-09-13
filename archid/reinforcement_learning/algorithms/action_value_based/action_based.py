from ..algorithm import Algorithm


class ActionBased(Algorithm):

    def available_actions(self, state):
        """Return configured legal actions for a state."""
        actions = getattr(self, "actions", None)
        if actions is None:
            raise ValueError("This algorithm needs an actions callable to select actions")
        return list(actions(state)) if callable(actions) else list(actions)

    def predict(self, state):
        """Select a deterministic greedy action (exploration belongs in train)."""
        actions = self.available_actions(state)
        if not actions:
            raise ValueError("No actions are available for the current state")
        return max(actions, key=lambda action: self.action_value(state, action))

    def action_value(self, state, action):
        raise NotImplementedError

    def td_target(
        self,
        reward,
        next_state,
        next_action,
        done
    ):
        if done:
            return reward

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
        done
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
