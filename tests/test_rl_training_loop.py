"""Regression tests for the high-level tabular RL training workflow."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from archid import (
    DoubleQLearning,
    ExpectedSARSA,
    MonteCarloControl,
    QLearning,
    ReinforcementLearning,
    SARSA,
)
from examples.cliff_walking_env import CliffWalking


def test_tabular_algorithms_train_through_orchestrator():
    actions = lambda state: [0, 1, 2, 3]
    algorithms = (QLearning, SARSA, ExpectedSARSA, DoubleQLearning, MonteCarloControl)
    for algorithm_class in algorithms:
        algorithm = algorithm_class(actions=actions)
        trainer = ReinforcementLearning(algorithm, CliffWalking())
        history = trainer.train(episodes=3, max_steps=20, epsilon=0.2, seed=4)
        assert len(history) == 3
        assert all(0 < item["steps"] <= 20 for item in history)
        assert isinstance(algorithm.predict(0), int)


if __name__ == "__main__":
    test_tabular_algorithms_train_through_orchestrator()
    print("RL training-loop tests passed")
