import euclid
import numpy as np
from archid.reinforcement_learning.algorithms.actor_critic.v_critic.v_critic import VCritic

class PPO(VCritic):

    def __init__(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs=10
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=gamma
        )

        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs

    def predict(self, state):
        distribution = self.actor.forward(state)

        return distribution.sample()

    def compute_advantages(
        self,
        rewards,
        values,
        next_values,
        dones
    ):

        advantages = []

        gae = euclid.Tensor(0.0)

        for t in reversed(range(len(rewards.data))):

            if dones[t]:

                delta = (
                    rewards[t]
                    - values[t]
                )

                gae = delta

            else:

                delta = (
                    rewards[t]
                    + self.gamma * next_values[t]
                    - values[t]
                )

                gae = (
                    delta
                    + self.gamma
                    * self.gae_lambda
                    * gae
                )

            advantages.append(
                gae.data.copy()
            )

        advantages.reverse()

        return euclid.Tensor(
            advantages
        )

    def update(self, experience):

        states = euclid.Tensor(
            np.stack([
                transition.state.data
                for transition in experience
            ])
        )

        actions = euclid.Tensor(
            np.asarray([
                transition.action
                for transition in experience
            ])
        )

        old_log_probs = euclid.Tensor(
            np.asarray([
                transition.log_prob
                for transition in experience
            ])
        )

        rewards = euclid.Tensor(
            np.asarray([
                transition.reward
                for transition in experience
            ])
        )

        next_states = euclid.Tensor(
            np.stack([
                transition.next_state.data
                for transition in experience
            ])
        )

        dones = [
            transition.done
            for transition in experience
        ]
        # --------------------------------------------------------
        # Compute value estimates
        # --------------------------------------------------------

        values = self.critic.forward(states)
        next_values = self.critic.forward(next_states)

        # --------------------------------------------------------
        # Compute GAE
        # --------------------------------------------------------

        advantages = self.compute_advantages(
            rewards,
            values,
            next_values,
            dones
        )

        # IMPORTANT:
        # Detach advantages from critic graph.
        advantages = euclid.Tensor(
            advantages.data.copy()
        )

        # --------------------------------------------------------
        # Returns
        # --------------------------------------------------------

        returns = values + advantages

        # IMPORTANT:
        # Detach returns from critic graph.
        returns = euclid.Tensor(
            returns.data.copy()
        )

        # --------------------------------------------------------
        # Normalize advantages
        # --------------------------------------------------------

        mean = advantages.mean()
        centered = advantages - mean
        variance = (centered * centered).mean()
        std = variance ** 0.5

        advantages = centered / (std + 1e-8)

        # Detach again so actor graph doesn't connect to critic.
        advantages = euclid.Tensor(
            advantages.data.copy()
        )

        # --------------------------------------------------------
        # PPO epochs
        # --------------------------------------------------------
        for _ in range(self.epochs):

            # ====================================================
            # ACTOR
            # ====================================================

            distribution = self.actor.forward(states)


            log_probs = distribution.log_prob(actions)
            # print("actions:", actions.data)
            # print("old_log_probs:", old_log_probs.data)
            # print("rewards:", rewards.data)
            # print("dones:", dones)
            ratio = (
                log_probs - old_log_probs
            ).exp()

            clipped_ratio = ratio.clip(
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon
            )
            advantages = euclid.Tensor(
                advantages.data.reshape(-1)
            )
            surrogate_1 = (
                ratio * advantages
            )

            surrogate_2 = (
                clipped_ratio * advantages
            )

            surrogate = surrogate_1.minimum(
                surrogate_2
            )

            entropy = distribution.entropy().mean()

            policy_loss = (
                -surrogate.mean()
                - self.entropy_coef * entropy
            )

            # Actor update
            self.actor_optimizer.zero_grad()

            policy_loss.backward()

            self.actor_optimizer.step()


            # ====================================================
            # CRITIC
            # ====================================================

            values = self.critic.forward(states)

            critic_loss = (
                (returns - values) ** 2
            ).mean()

            critic_loss = (
                self.value_coef * critic_loss
            )

            # Critic update
            self.critic_optimizer.zero_grad()

            critic_loss.backward()

            self.critic_optimizer.step()