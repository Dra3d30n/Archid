from ..learning import Learning


class ReinforcementLearning(Learning):

    def __init__(
        self,
        algorithm,
        policy,
        optimizer
    ):
        super().__init__()

        self.algorithm = algorithm
        self.policy = policy
        self.optimizer = optimizer
    def attach_network(self, network):
        self.optimizer.set_parameters(network.parameters())
        return super().attach_network(network)
    def predict(self, state):
        return self.policy.sample_action(
            self.network.forward(state)
        )

    def update_model(self, **data):
        update = getattr(self.algorithm, "update", None)
        if update is not None:
            return update(
                self.policy,
                self.optimizer,
                **data
            )

        loss = self.algorithm.compute_loss(
            self.policy,
            **data
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss
