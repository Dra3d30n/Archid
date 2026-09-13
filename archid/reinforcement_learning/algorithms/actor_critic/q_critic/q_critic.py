
from ..actor_critic import ActorCritic


class QCritic(ActorCritic):

    def __init__(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        gamma=0.99
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer
        )

        self.gamma = gamma

    def q_value(self, state, action):
        return self.critic.forward(
            state,
            action
        )

    def td_target(
        self,
        reward,
        next_state,
        next_action,
        done
    ):
        if done:
            return reward

        next_value = self.q_value(
            next_state,
            next_action
        )

        return reward + self.gamma * float(
            next_value.data.reshape(-1)[0]
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
        q_value = self.q_value(
            state,
            action
        )

        return self.td_target(
            reward,
            next_state,
            next_action,
            done
        ) - q_value

    def update(self, experience):
        raise NotImplementedError
