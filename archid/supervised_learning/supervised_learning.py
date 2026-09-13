from euclid.core.tensor import Tensor


class SupervisedLearning():

    name = "Supervised"

    def __init__(self, loss, optimizer, network):

        self.loss = loss
        self.optimizer = optimizer
        self.network = network
        self.optimizer.set_parameters(network.parameters())




    def train(self, loader, steps):
        self.network.train()
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
        self.network.eval()