from ..learning import Learning
from euclid.core.tensor import Tensor


class SupervisedLearning(Learning):

    name = "Supervised"

    def __init__(self, loss, optimizer):
        super().__init__()

        self.loss = loss
        self.optimizer = optimizer

    def attach_network(self, network):
        self.optimizer.set_parameters(
            network.parameters()
        )
        return super().attach_network(network)

    def train(self, loader, steps):

        for step in range(steps):

            x_batch, y_batch = loader.batch()

            if not isinstance(x_batch, Tensor):
                x_batch = Tensor(x_batch)

            if not isinstance(y_batch, Tensor):
                y_batch = Tensor(y_batch)

            self.optimizer.zero_grad()

            prediction = self.network.forward(
                x_batch
            )

            loss = self.loss.forward(
                prediction,
                y_batch
            )


            loss.backward()

            self.optimizer.step()



            print(
                f"Step {step + 1}/{steps} | "
                f"Loss: {float(loss.data):.6f}",
                flush=True,
            )            
            del prediction
            del loss
            del x_batch
            del y_batch