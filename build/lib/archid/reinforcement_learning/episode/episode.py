from euclid.settings import xp


class Step:

    def __init__(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        log_probability
    ):

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

        # Stored from behavior policy during rollout
        self.log_probability = (log_probability)
        self.old_log_probability = (log_probability)

        self.reward_to_go = None
        self.value = None
        self.advantage = None


class Episode:

    def __init__(self):
        self.steps = []


    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        log_probability
    ):

        self.steps.append(
            Step(
                state,
                action,
                reward,
                next_state,
                done,
                log_probability
            )
        )


    def discounted_returns(self, gamma):

        returns = xp.empty(
            len(self.steps),
            dtype=xp.float32
        )

        running_return = 0.0

        for index in range(len(self.steps)-1, -1, -1):

            step = self.steps[index]

            running_return = (
                float(step.reward)
                +
                gamma *
                running_return *
                (not step.done)
            )

            returns[index] = running_return
            step.reward_to_go = running_return

        return returns


    def set_advantages(
        self,
        gamma,
        normalize=False
    ):

        returns = self.discounted_returns(gamma)

        advantages = returns.copy()

        if normalize and len(advantages) > 1:

            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std() + 1e-8
            )

        for step, advantage in zip(
            self.steps,
            advantages
        ):
            step.advantage = float(advantage)

        return advantages


    def set_returns(self, gamma):

        returns = self.discounted_returns(gamma)

        for step, value in zip(
            self.steps,
            returns
        ):
            step.reward_to_go = float(value)

        return returns