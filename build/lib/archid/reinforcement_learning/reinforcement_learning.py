from .experience import Experience, Transition


class ReinforcementLearning:

    def __init__(
        self,
        algorithm,
        environment
    ):

        self.algorithm = algorithm
        self.environment = environment

    def predict(self, state):
        return self.algorithm.predict(state)

    def update_model(self, experience):
        return self.algorithm.update(experience)
    def train(self,epochs):
        for epoch in epochs:
            experience=self.run()
            self.update_model(experience)
    def run(self, steps=None):

        experience = Experience()

        state = self.environment.reset()

        step = 0

        while True:

            action = self.predict(state)

            next_state, reward, done = (
                self.environment.step(action)
            )

            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            experience.add(transition)

            state = next_state
            step += 1

            if done:
                break

            if steps is not None and step >= steps:
                break

        return experience
