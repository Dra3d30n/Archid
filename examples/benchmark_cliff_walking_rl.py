"""
Benchmark all RL algorithms implemented by Archid on Cliff Walking.

Algorithms:
    QLearning, SARSA, ExpectedSARSA, DoubleQLearning,
    MonteCarloControl, DQN, QActorCritic, PPO

Training is handled through Archid's high-level
ReinforcementLearning.train() API.

The benchmark also saves the learned greedy solution
(one action for every Cliff Walking state).

Run:
    MPLCONFIGDIR=/tmp/archid-mpl python examples/benchmark_cliff_walking_rl.py
"""

import csv
import random
import sys
from pathlib import Path

import euclid
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from archid import (
    DQN,
    PPO,
    QActorCritic,
    QLearning,
    SARSA,
    ExpectedSARSA,
    DoubleQLearning,
    MonteCarloControl,
    ReinforcementLearning,
)

from examples.cliff_walking_env import CliffWalking
from examples.train_ppo_cliff_walking import encode_state as ppo_encode_state
from examples.train_ppo_cliff_walking import train_ppo


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALGORITHMS = (
    "QLearning",
    "SARSA",
    "ExpectedSARSA",
    "DoubleQLearning",
    "MonteCarloControl",
    "DQN",
    "QActorCritic",
    "PPO",
)

SEEDS = range(1)
NEURAL_SEEDS = range(1)

EPISODES = {
    "QLearning": 500,
    "SARSA": 500,
    "ExpectedSARSA": 3000,
    "DoubleQLearning": 3000,
    "MonteCarloControl": 3000,
    "DQN": 1000,
    "QActorCritic": 1500,
    "PPO": 300,
}

MAX_STEPS = 100
EPSILON = 0.10

STATE_COUNT = 48
ACTION_COUNT = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_state(state):
    """
    Encode a Cliff Walking state ID as a one-hot vector.

    Cliff Walking has 48 states, so state N becomes a vector
    of length 48 with a 1 at index N.
    """
    encoded = np.zeros(
        STATE_COUNT,
        dtype=np.float32,
    )

    encoded[int(state)] = 1.0

    return encoded


def scalar(value):
    """
    Convert either a Python scalar or an Euclid scalar Tensor
    into a Python float.
    """
    if hasattr(value, "data"):
        return float(
            value.data.reshape(-1)[0]
        )

    return float(value)


def record(
    algorithm,
    seed,
    episode,
    total,
    steps,
):
    return {
        "algorithm": algorithm,
        "seed": seed,
        "episode": episode + 1,
        "return": total,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Encoded environment
# ---------------------------------------------------------------------------

class EncodedCliffWalking:
    """
    Wrapper around CliffWalking that converts integer state IDs
    into one-hot vectors for neural networks.

    The underlying environment still uses integer states.
    """

    def __init__(self):
        self.env = CliffWalking()

    def reset(self):
        state = self.env.reset()
        return encode_state(state)

    def step(self, action):
        state, reward, done = self.env.step(action)

        return (
            encode_state(state),
            reward,
            done,
        )

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------

class DiscreteActor:
    """
    Euclid actor producing a four-action probability vector.
    """

    def __init__(self):
        self.network = euclid.networks.Sequential([
            euclid.layers.Dense(48, 32),
            euclid.layers.activations.Tanh(),
            euclid.layers.Dense(32, 4),
            euclid.layers.activations.Softmax(),
        ])

    def forward(self, state):
        if not isinstance(
            state,
            euclid.Tensor,
        ):
            state = euclid.Tensor(
                np.asarray(
                    state,
                    dtype=np.float32,
                ).reshape(1, -1)
            )

        return self.network.forward(state)

    def parameters(self):
        return self.network.parameters()


class DiscreteQCritic:
    """
    Q(s,a) critic compatible with Archid DQN
    and QActorCritic.
    """

    def __init__(self):
        self.network = euclid.networks.Sequential([
            euclid.layers.Dense(48, 32),
            euclid.layers.activations.Tanh(),
            euclid.layers.Dense(32, 4),
        ])

    def forward(self, state, action):
        if not isinstance(
            state,
            euclid.Tensor,
        ):
            state = euclid.Tensor(
                np.asarray(
                    state,
                    dtype=np.float32,
                ).reshape(1, -1)
            )

        q_values = self.network.forward(state)

        if isinstance(
            action,
            euclid.Tensor,
        ):
            action_vector = action

        else:
            action_vector = np.zeros(
                (
                    q_values.data.shape[0],
                    ACTION_COUNT,
                ),
                dtype=np.float32,
            )

            action_vector[
                :,
                int(action)
            ] = 1.0

            action_vector = euclid.Tensor(
                action_vector
            )

        return (
            q_values * action_vector
        ).sum(axis=-1)

    def parameters(self):
        return self.network.parameters()


def make_optimizer(
    parameters,
    learning_rate=0.001,
):
    optimizer = euclid.optimizers.Adam(
        learning_rate=learning_rate
    )

    optimizer.set_parameters(
        parameters
    )

    return optimizer


# ---------------------------------------------------------------------------
# Tabular algorithms
# ---------------------------------------------------------------------------

def train_tabular(
    name,
    seed,
    episodes,
):
    """
    Train a tabular algorithm using the high-level
    ReinforcementLearning.train() API.
    """

    env = CliffWalking()

    classes = {
        "QLearning": QLearning,
        "SARSA": SARSA,
        "ExpectedSARSA": ExpectedSARSA,
        "DoubleQLearning": DoubleQLearning,
    }

    if name == "MonteCarloControl":
        agent = MonteCarloControl(
            actions=lambda _: [
                0,
                1,
                2,
                3,
            ],
            gamma=0.99,
            learning_rate=0.1,
        )

    else:
        agent = classes[name](
            actions=lambda _: [
                0,
                1,
                2,
                3,
            ],
            gamma=0.99,
            learning_rate=0.1,
        )

    # Monte Carlo needs the completed episode.
    #
    # All other algorithms default to step-based updates.
    if name == "MonteCarloControl":
        agent.learning_mode = "episode"

    else:
        agent.learning_mode = "step"

    trainer = ReinforcementLearning(
        agent,
        env,
    )

    history = trainer.train(
        episodes=episodes,
        max_steps=MAX_STEPS,
        epsilon=EPSILON,
        seed=seed,
    )

    rows = [
        record(
            name,
            seed,
            row["episode"] - 1,
            row["return"],
            row["steps"],
        )
        for row in history
    ]

    def policy(state):
        actions = agent.available_actions(
            state
        )

        return max(
            actions,
            key=lambda action: scalar(
                agent.action_value(
                    state,
                    action,
                )
            ),
        )

    return rows, policy


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

def train_dqn(
    seed,
    episodes,
):
    """
    Train DQN using ReinforcementLearning.train().
    """

    np.random.seed(seed)

    env = EncodedCliffWalking()

    critic = DiscreteQCritic()

    agent = DQN(
        critic,
        make_optimizer(
            critic.parameters(),
            0.01,
        ),
        actions=[
            0,
            1,
            2,
            3,
        ],
        gamma=0.99,
    )

    agent.learning_mode = "step"

    trainer = ReinforcementLearning(
        agent,
        env,
    )

    history = trainer.train(
        episodes=episodes,
        max_steps=MAX_STEPS,
        epsilon=EPSILON,
        seed=seed,
    )

    rows = [
        record(
            "DQN",
            seed,
            row["episode"] - 1,
            row["return"],
            row["steps"],
        )
        for row in history
    ]

    def policy(state):
        encoded = encode_state(
            state
        )

        values = [
            scalar(
                agent.action_value(
                    encoded,
                    action,
                )
            )
            for action in range(
                ACTION_COUNT
            )
        ]

        return int(
            np.argmax(values)
        )

    return rows, policy


# ---------------------------------------------------------------------------
# Q Actor-Critic
# ---------------------------------------------------------------------------

def train_q_actor_critic(
    seed,
    episodes,
):
    """
    Train QActorCritic using ReinforcementLearning.train().
    """

    np.random.seed(seed)

    env = EncodedCliffWalking()

    actor = DiscreteActor()
    critic = DiscreteQCritic()

    agent = QActorCritic(
        actor,
        critic,
        make_optimizer(
            actor.parameters(),
            0.001,
        ),
        make_optimizer(
            critic.parameters(),
            0.01,
        ),
        gamma=0.99,
    )

    agent.learning_mode = "step"

    trainer = ReinforcementLearning(
        agent,
        env,
    )

    history = trainer.train(
        episodes=episodes,
        max_steps=MAX_STEPS,
        epsilon=EPSILON,
        seed=seed,
    )

    rows = [
        record(
            "QActorCritic",
            seed,
            row["episode"] - 1,
            row["return"],
            row["steps"],
        )
        for row in history
    ]

    def policy(state):
        output = actor.forward(
            encode_state(state)
        )

        return int(
            output.data
            .reshape(-1)
            .argmax()
        )

    return rows, policy


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def train_ppo_archid(
    seed,
    episodes,
):
    """
    PPO remains delegated to the existing PPO training
    implementation because train_ppo() currently owns
    PPO-specific rollout and optimization behavior.
    """

    random.seed(seed)
    np.random.seed(seed)

    agent, _, rewards = train_ppo(
        episodes=episodes
    )

    rows = [
        record(
            "PPO",
            seed,
            episode,
            value,
            "",
        )
        for episode, value in enumerate(
            rewards
        )
    ]

    def policy(state):
        output = agent.actor.forward(
            euclid.Tensor(
                ppo_encode_state(
                    state
                ).reshape(1, -1)
            )
        )

        return int(
            output._softmax()
            .data
            .reshape(-1)
            .argmax()
        )

    return rows, policy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(policy):
    """
    Evaluate a greedy policy without exploration.
    """

    env = CliffWalking()

    returns = []

    for _ in range(50):
        state = env.reset()
        total = 0.0

        for _ in range(MAX_STEPS):
            action = policy(state)

            state, reward, done = (
                env.step(action)
            )

            total += reward

            if done:
                break

        returns.append(total)

    return float(
        np.mean(returns)
    )


# ---------------------------------------------------------------------------
# Solution saving
# ---------------------------------------------------------------------------

def save_solution(
    output,
    algorithm,
    policy,
):
    """
    Save the greedy policy for every Cliff Walking state.
    """

    rows = []

    for state in range(
        STATE_COUNT
    ):
        action = int(
            policy(state)
        )

        rows.append({
            "algorithm": algorithm,
            "state": state,
            "action": action,
        })

    path = (
        output
        / f"{algorithm.lower()}_solution.csv"
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "state",
                "action",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def save_solution_summary(
    output,
    solution_rows,
):
    path = (
        output
        / "solutions.csv"
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "state",
                "action",
            ],
        )

        writer.writeheader()
        writer.writerows(
            solution_rows
        )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    output = (
        ROOT
        / "examples"
        / "results"
        / "cliff_walking_rl"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    summary = []
    all_solutions = []

    tabular = {
        "QLearning",
        "SARSA",
        "ExpectedSARSA",
        "DoubleQLearning",
        "MonteCarloControl",
    }

    trainers = {
        "DQN": train_dqn,
        "QActorCritic": train_q_actor_critic,
        "PPO": train_ppo_archid,
    }

    for name in ALGORITHMS:
        evaluations = []
        episodes = EPISODES[name]

        seeds = (
            SEEDS
            if name in tabular
            else NEURAL_SEEDS
        )

        for seed in seeds:

            if name in tabular:
                trial_rows, policy = (
                    train_tabular(
                        name,
                        seed,
                        episodes,
                    )
                )

            else:
                trial_rows, policy = (
                    trainers[name](
                        seed,
                        episodes,
                    )
                )

            rows.extend(
                trial_rows
            )

            evaluation = evaluate(
                policy
            )

            evaluations.append(
                evaluation
            )

            solution = save_solution(
                output,
                name,
                policy,
            )

            all_solutions.extend(
                solution
            )

        mean_return = float(
            np.mean(evaluations)
        )

        std_return = float(
            np.std(
                evaluations,
                ddof=0,
            )
        )

        summary.append({
            "algorithm": name,
            "seeds": len(evaluations),
            "training_episodes": episodes,
            "mean_eval_return": mean_return,
            "std_eval_return": std_return,
        })

        print(
            f"{name:18s} "
            f"{mean_return:7.2f}"
        )

    # -----------------------------------------------------------------------
    # Episode results
    # -----------------------------------------------------------------------

    with (
        output / "episode_results.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        fields = list(
            rows[0]
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    with (
        output / "summary.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        fields = list(
            summary[0]
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(summary)

    # -----------------------------------------------------------------------
    # Combined solutions
    # -----------------------------------------------------------------------

    save_solution_summary(
        output,
        all_solutions,
    )

    # -----------------------------------------------------------------------
    # Learning curves
    # -----------------------------------------------------------------------

    def moving_average(values, window=50):
        values = np.asarray(values, dtype=float)

        if len(values) < window:
            return values

        kernel = np.ones(window) / window

        return np.convolve(
            values,
            kernel,
            mode="valid",
        )

    def plot_group(names, filename, title):
        plt.figure(figsize=(12, 7))

        for name in names:
            subset = [
                row
                for row in rows
                if row["algorithm"] == name
            ]

            subset.sort(
                key=lambda row: row["episode"]
            )

            values = np.asarray([
                row["return"]
                for row in subset
            ], dtype=float)

            episodes = np.arange(
                1,
                len(values) + 1,
            )

            # Raw training return
            plt.plot(
                episodes,
                values,
                alpha=0.12,
                linewidth=0.8,
            )

            # Smoothed return
            window = min(
                50,
                len(values),
            )

            smoothed = moving_average(
                values,
                window,
            )

            smooth_x = np.arange(
                window,
                len(values) + 1,
            )

            plt.plot(
                smooth_x,
                smoothed,
                linewidth=2,
                label=name,
            )

        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Return")

        plt.grid(
            alpha=0.25,
            linestyle="--",
        )

        plt.legend(
            loc="best",
            frameon=True,
        )

        plt.tight_layout()

        plt.savefig(
            output / filename,
            dpi=220,
            bbox_inches="tight",
        )

        plt.close()

    # Tabular algorithms
    plot_group(
        [
            "QLearning",
            "SARSA",
            "ExpectedSARSA",
            "DoubleQLearning",
            "MonteCarloControl",
        ],
        "learning_curves_tabular.png",
        "Archid RL — Tabular Algorithms on Cliff Walking",
    )

    # Neural algorithms
    plot_group(
        [
            "DQN",
            "QActorCritic",
            "PPO",
        ],
        "learning_curves_neural.png",
        "Archid RL — Neural Algorithms on Cliff Walking",
    )

    # Combined comparison using normalized training progress.
    #
    # This makes algorithms with different episode counts comparable.
    plt.figure(figsize=(12, 7))

    for name in ALGORITHMS:
        subset = [
            row
            for row in rows
            if row["algorithm"] == name
        ]

        subset.sort(
            key=lambda row: row["episode"]
        )

        values = np.asarray([
            row["return"]
            for row in subset
        ], dtype=float)

        if len(values) < 2:
            continue

        window = min(
            50,
            len(values),
        )

        smoothed = moving_average(
            values,
            window,
        )

        # Convert to percentage of training completed.
        progress = np.linspace(
            0,
            100,
            len(smoothed),
        )

        plt.plot(
            progress,
            smoothed,
            linewidth=2,
            label=name,
        )

    plt.title(
        "Archid RL — Training Progress Comparison"
    )

    plt.xlabel(
        "Training progress (%)"
    )

    plt.ylabel(
        "Return"
    )

    plt.grid(
        alpha=0.25,
        linestyle="--",
    )

    plt.legend(
        loc="best",
        frameon=True,
    )

    plt.tight_layout()

    plt.savefig(
        output / "learning_curves_normalized.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

if __name__ == "__main__":
    main()

