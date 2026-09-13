
from .q_critic import QCritic


class QActorCritic(QCritic):

    def predict(self, state):
        return self.actor.forward(state)
    def expected_q_value(self, state):

        probs = self.actor.forward(state)
        probs = probs.reshape(-1)

        action_dim = probs.data.shape[0]

        total = None

        for action in range(action_dim):

            q_value = self.q_value(
                state,
                action
            )

            q_value = float(
                q_value.data.reshape(-1)[0]
            )

            contribution = probs[action] * q_value

            if total is None:
                total = contribution
            else:
                total = total + contribution

        return total

    def expected_q_target(self, next_state):
        """
        Calculate:

            E[Q(s',a')] = sum_a pi(a'|s') Q(s',a')

        This returns a Python float so the TD target is detached
        from both the actor and critic computation graphs.
        """

        probs = self.actor.forward(next_state)

        action_dim = probs.data.shape[-1]

        total = 0.0

        for action in range(action_dim):

            probability = float(
                probs.data.reshape(-1)[action]
            )

            q_value = self.q_value(
                next_state,
                action
            )

            q_value = float(
                q_value.data.reshape(-1)[0]
            )

            total += probability * q_value

        return total

    def update(self, experience):

        critic_losses = []
        actor_losses = []

        for transition in experience:

            state = transition.state
            action = transition.action
            reward = transition.reward
            next_state = transition.next_state
            done = transition.done

            # -------------------------------------------------
            # Critic update
            # -------------------------------------------------

            current_q = self.q_value(
                state,
                action
            )

            if done:
                target = reward

            else:
                next_value = self.expected_q_target(
                    next_state
                )

                target = (
                    reward
                    + self.gamma * next_value
                )

            td_error = target - current_q

            critic_loss = td_error ** 2

            self.critic_optimizer.zero_grad()

            critic_loss.backward()

            self.critic_optimizer.step()

            # -------------------------------------------------
            # Actor update
            # -------------------------------------------------

            expected_q = self.expected_q_value(
                state
            )

            actor_loss = -expected_q

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            self.actor_optimizer.step()

            critic_losses.append(
                critic_loss
            )

            actor_losses.append(
                actor_loss
            )

        return {
            "critic_loss": critic_losses,
            "actor_loss": actor_losses
        }
