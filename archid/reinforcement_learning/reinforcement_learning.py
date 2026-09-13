
"""High-level training loop for Archid reinforcement-learning algorithms."""

import random

from .experience import Experience, Transition


class ReinforcementLearning:
    """Pair an algorithm and environment, then train them with one method."""

    def __init__(self, algorithm, environment):
        self.algorithm = algorithm
        self.environment = environment

    def predict(self, state):
        return self.algorithm.predict(state)

    def _select_action(self, state, epsilon, rng):
        if (
            hasattr(self.algorithm, "action_value")
            and getattr(self.algorithm, "actions", None) is not None
        ):
            actions = self.algorithm.available_actions(state)

            if rng.random() < epsilon:
                return rng.choice(actions)

            def value(action):
                result = self.algorithm.action_value(state, action)

                if hasattr(result, "data"):
                    return float(result.data.reshape(-1)[0])

                return result

            return max(actions, key=value)

        return self.predict(state)

    def train(
        self,
        episodes,
        max_steps=None,
        epsilon=0.1,
        seed=None,
    ):
        """Train for a number of episodes.

        Algorithms can specify how experience is consumed through
        ``update_mode``:

        ``"step"``
            Update after every transition.

        ``"episode"``
            Collect the complete episode, then update once.

        The default is ``"step"``.
        """
        if episodes < 1:
            raise ValueError("episodes must be positive")

        rng = random.Random(seed)
        history = []

        update_mode = getattr(
            self.algorithm,
            "update_mode",
            "step",
        )

        if update_mode not in {"step", "episode"}:
            raise ValueError(
                f"Invalid update_mode: {update_mode!r}. "
                "Expected 'step' or 'episode'."
            )

        for episode in range(episodes):
            state = self.environment.reset()

            total_reward = 0.0
            steps = 0
            done = False

            experience = Experience()

            action = self._select_action(
                state,
                epsilon,
                rng,
            )

            while not done and (
                max_steps is None or steps < max_steps
            ):
                next_state, reward, done = (
                    self.environment.step(action)
                )

                steps += 1
                total_reward += reward

                next_action = (
                    None
                    if done
                    else self._select_action(
                        next_state,
                        epsilon,
                        rng,
                    )
                )

                transition = Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    next_action,
                )

                experience.add(transition)

                # TD / online algorithms learn immediately.
                if update_mode == "step":
                    step_experience = Experience()
                    step_experience.add(transition)

                    self.algorithm.update(
                        step_experience
                    )

                state = next_state
                action = next_action

            # Monte Carlo and other episode-based algorithms
            # receive the entire trajectory.
            if update_mode == "episode":
                self.algorithm.update(experience)

            history.append({
                "episode": episode + 1,
                "return": total_reward,
                "steps": steps,
                "done": done,
            })

        return history

    def run(self, steps=None):
        """Collect one greedy episode without updating the algorithm."""

        state = self.environment.reset()
        experience = Experience()

        count = 0
        done = False

        while not done and (
            steps is None or count < steps
        ):
            action = self.predict(state)

            next_state, reward, done = (
                self.environment.step(action)
            )

            experience.add(
                Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
            )

            state = next_state
            count += 1

        return experience
