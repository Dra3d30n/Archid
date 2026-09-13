
from ..algorithm import Algorithm


class ActorCritic(Algorithm):

    def __init__(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer
    ):
        self.actor = actor
        self.critic = critic

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def predict(self, state):
        action = self.actor.forward(state)
        return action

    def update(self, experience):
        raise NotImplementedError
