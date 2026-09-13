from ..evolutionary_learning import EvolutionaryLearning
from euclid.core.tensor import Tensor


class EvolutionarySupervisedLearning(EvolutionaryLearning):
    name = "EvolutionarySupervised"

    def __init__(
        self,
        loss,
        mutation_rate,
        population_size,
        selection_size
    ):
        super().__init__(
            mutation_rate,
            population_size,
            selection_size
        )

        self.loss = loss

    def train(self, loader, epochs):
        self.population=[]
        self.initialize_population()
        for epoch in range(epochs):

            # Evaluate population
            fitness = []

            for network in self.population:
                total_loss = 0

                for x, y in loader:
                    x = Tensor(x)
                    y = Tensor(y)

                    prediction = network(x)
                    loss = self.loss(prediction, y)

                    total_loss += loss.data

                fitness.append(-total_loss)

            # Evolution step
            self.select(fitness)
            self.reproduce()
            self.mutate()
            best_fitness = max(fitness)
