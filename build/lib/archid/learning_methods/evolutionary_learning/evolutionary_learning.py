from ..learning import Learning
import copy
import numpy as np

class EvolutionaryLearning(Learning):
    name = "Evolutionary"

    def __init__(self, mutation_rate, population_size, selection_size):
        super().__init__()

        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.selection_size = selection_size

        self.population = []

    def reproduce(self):
        new_population = []

        # Keep elites unchanged
        new_population.extend(self.population)

        # Fill the rest with copies
        while len(new_population) < self.population_size:
            parent = np.random.choice(self.population)
            child = copy.deepcopy(parent)
            new_population.append(child)

        self.population = new_population
    def select(self, fitness):
        # same for SL and RL
        self.population = sorted(
            self.population,
            key=lambda x: fitness[self.population.index(x)],
            reverse=True
        )[:self.selection_size]
    def initialize_population(self):
        """
        Creates the initial population from a starting network.
        """

        self.population = []

        for _ in range(self.population_size):
            individual = copy.deepcopy(self.network)
            self.population.append(individual)

        return self.population
    def mutate(self):
        for i, network in enumerate(self.population):

            # Skip elites
            if i < self.selection_size:
                continue

            for layer in network.layers:

                if hasattr(layer, "weights"):
                    weights = layer.weights.data

                    mask = np.random.random(weights.shape) < self.mutation_rate
                    noise = np.random.normal(0, 0.01, weights.shape)

                    weights += mask * noise

                if hasattr(layer, "bias"):
                    bias = layer.bias.data

                    mask = np.random.random(bias.shape) < self.mutation_rate
                    noise = np.random.normal(0, 0.01, bias.shape)

                    bias += mask * noise