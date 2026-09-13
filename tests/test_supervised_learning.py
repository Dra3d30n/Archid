import sys
from pathlib import Path
import unittest

import euclid as Euclid
from euclid import MSE, Tensor
from euclid.backend import xp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from archid.supervised_learning.supervised_learning import (
    SupervisedLearning,
)
from archid.model.model import Model


class SupervisedLearningTests(unittest.TestCase):
    def test_training_learns_linear_mapping(self):
        xp.random.seed(11)
        inputs = xp.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        targets = 3.0 * inputs - 0.5
        network = Euclid.Sequential([Euclid.Dense(1, 1)])
        learning = SupervisedLearning(MSE(), Euclid.Adam(learning_rate=0.05))
        Model(network, learning)

        initial_loss = float(MSE().forward(network.forward(Tensor(inputs)), Tensor(targets)).data)
        learning.train([(inputs, targets)], epochs=150)
        final_loss = float(MSE().forward(network.forward(Tensor(inputs)), Tensor(targets)).data)

        self.assertLess(final_loss, initial_loss * 0.01)


if __name__ == "__main__":
    unittest.main()
