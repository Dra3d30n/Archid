"""
Simple smoke-test script that attempts to instantiate and call `update` on
available RL algorithms. Prints success/failure for each algorithm.
"""
from pprint import pprint
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Algorithm classes
from archid import QLearning, SARSA, DQN, TD, PPO
from archid.reinforcement_learning.experience.experience import (
    Experience,
    Transition,
)


class DummyCritic:
    def forward(self, *args, **kwargs):
        # return a scalar-like value
        return 0.0

    def parameters(self):
        return []


class SimpleOptimizer:
    def set_parameters(self, params):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass


class DummyEnv:
    def reset(self):
        return 0

    def step(self, action):
        return 0, 0.0, True


def make_experience():
    exp = Experience()
    t = Transition(state=0, action=0, reward=0.0, next_state=0, done=True)
    exp.add(t)
    return exp


def try_update(algo, experience):
    # try multiple common call styles
    for call in (lambda a: algo.update(a), lambda a: algo.update(experience=a), lambda a: algo.update(episode=a)):
        try:
            return True, call(experience)
        except TypeError:
            continue
        except Exception as e:
            return False, e
    return False, TypeError("no compatible update signature")


def main():
    results = {}

    safe_ctors = {
        "QLearning": lambda: QLearning(actions=lambda s: [0, 1], gamma=0.99),
        "SARSA": lambda: SARSA(gamma=0.99, learning_rate=0.1),
        "TD": lambda: TD(learning_rate=0.1, gamma=0.99),
        "DQN": lambda: DQN(critic=DummyCritic(), critic_optimizer=SimpleOptimizer(), actions=[0, 1], gamma=0.99),
        "PPO": lambda: PPO(gamma=0.99),
    }

    experience = make_experience()

    for name, ctor in safe_ctors.items():
        try:
            algo = ctor()
        except Exception as e:
            results[name] = {"instantiated": False, "error": repr(e)}
            continue

        ok, out = try_update(algo, experience)
        results[name] = {"instantiated": True, "update_ok": ok, "result": repr(out)}

    pprint(results)


if __name__ == "__main__":
    main()
