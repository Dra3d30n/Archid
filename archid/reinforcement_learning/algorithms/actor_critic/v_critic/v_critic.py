from ..actor_critic import ActorCritic


class VCritic(ActorCritic):

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

    def td_target(self, reward, next_state, done):
        next_value = self.critic.forward(next_state)

        if done:
            return reward

        return reward + self.gamma * next_value

    def td_error(
        self,
        state,
        reward,
        next_state,
        done
    ):
        value = self.critic.forward(state)

        return self.td_target(
            reward,
            next_state,
            done
        ) - value

    def update(self, experience):
        raise NotImplementedError
