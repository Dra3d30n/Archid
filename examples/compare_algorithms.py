"""
Compare Q-Learning vs PPO on Cliff Walking.

Shows side-by-side performance comparison and visualizations.
"""

import sys
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import euclid
print(euclid.__file__)
from archid import QLearning
from archid.reinforcement_learning.algorithms.actor_critic.v_critic.ppo import PPO
from archid.reinforcement_learning.experience import (
    Experience,
    Transition
)
from examples.cliff_walking_env import CliffWalking


# ============================================================
# Configuration
# ============================================================

STATE_DIM = 48
ACTION_DIM = 4
HIDDEN_DIM = 128

PPO_ACTOR_LR = 3e-4
PPO_CRITIC_LR = 1e-3

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
PPO_EPOCHS = 10


# ============================================================
# Categorical Distribution
# ============================================================

class CategoricalDistribution:

    def __init__(self, logits):
        self.logits = logits

    def _softmax(self):

        shifted = (
            self.logits
            - self.logits.max(
                axis=-1,
                keepdims=True
            )
        )

        exp_logits = shifted.exp()

        return exp_logits / (
            exp_logits.sum(
                axis=-1,
                keepdims=True
            ) + 1e-8
        )

    def sample(self):

        probs = self._softmax()

        probs_np = np.asarray(
            probs.data
        )

        if probs_np.ndim == 2:
            probs_np = probs_np[0]

        return int(
            np.random.choice(
                ACTION_DIM,
                p=probs_np
            )
        )

    def log_prob(self, actions):

        actions_np = np.asarray(
            actions.data
            if isinstance(actions, euclid.Tensor)
            else actions
        ).astype(np.int64)

        actions_np = actions_np.reshape(-1)

        # log softmax:
        #
        # log P(a) = logits[a] - log(sum(exp(logits)))
        #

        max_logits = self.logits.max(
            axis=-1,
            keepdims=True
        )

        shifted = (
            self.logits - max_logits
        )

        log_sum_exp = (
            shifted.exp().sum(
                axis=-1,
                keepdims=True
            )
        ).log()

        log_probs = (
            shifted - log_sum_exp
        )

        # Constant one-hot selector.
        one_hot = np.zeros(
            (
                len(actions_np),
                self.logits.data.shape[-1]
            ),
            dtype=np.float32
        )

        for i, action in enumerate(actions_np):
            one_hot[i, int(action)] = 1.0

        one_hot = euclid.Tensor(one_hot)

        return (
            log_probs * one_hot
        ).sum(axis=-1)
    def entropy(self):

        probs = self._softmax()

        return -(
            probs
            * (probs + 1e-8).log()
        ).sum(axis=-1)
# ============================================================
# Actor
# ============================================================

class Actor:

    def __init__(
        self,
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM
    ):

        self.network = euclid.networks.Sequential([
            euclid.layers.Dense(
                state_dim,
                hidden_dim
            ),

            euclid.layers.activations.Tanh(),

            euclid.layers.Dense(
                hidden_dim,
                hidden_dim
            ),

            euclid.layers.activations.Tanh(),

            euclid.layers.Dense(
                hidden_dim,
                action_dim
            )
        ])

    def forward(self, state):

        logits = self.network.forward(
            state
        )

        return CategoricalDistribution(
            logits
        )

    def parameters(self):

        return self.network.parameters()


# ============================================================
# Critic
# ============================================================

class Critic:

    def __init__(
        self,
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM
    ):

        self.network = euclid.networks.Sequential([
            euclid.layers.Dense(
                state_dim,
                hidden_dim
            ),

            euclid.layers.activations.Tanh(),

            euclid.layers.Dense(
                hidden_dim,
                hidden_dim
            ),

            euclid.layers.activations.Tanh(),

            euclid.layers.Dense(
                hidden_dim,
                1
            )
        ])

    def forward(self, state):

        return self.network.forward(
            state
        )

    def parameters(self):

        return self.network.parameters()


# ============================================================
# State Encoding
# ============================================================

def encode_state(state):

    encoded = np.zeros(
        STATE_DIM,
        dtype=np.float32
    )

    encoded[state] = 1.0

    return encoded


# ============================================================
# Train PPO
# ============================================================

def train_ppo(episodes=1000):
    """Train PPO using the actual Archid PPO implementation."""

    env = CliffWalking()

    # --------------------------------------------------------
    # Networks
    # --------------------------------------------------------

    actor = Actor()
    critic = Critic()

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    actor_optimizer = euclid.optimizers.Adam(
        learning_rate=PPO_ACTOR_LR
    )

    actor_optimizer.set_parameters(
        actor.parameters()
    )

    critic_optimizer = euclid.optimizers.Adam(
        learning_rate=PPO_CRITIC_LR
    )

    critic_optimizer.set_parameters(
        critic.parameters()
    )

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    agent = PPO(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        epochs=PPO_EPOCHS
    )

    episode_rewards = []

    for episode in range(episodes):

        state = env.reset()

        episode_reward = 0
        done = False
        steps = 0

        experience = Experience()

        while not done and steps < 100:

            # ------------------------------------------------
            # Encode state
            # ------------------------------------------------

            state_encoded = encode_state(
                state
            )

            state_tensor = euclid.Tensor(
                state_encoded.reshape(1, -1)
            )

            # ------------------------------------------------
            # Get policy distribution
            # ------------------------------------------------

            distribution = actor.forward(
                state_tensor
            )

            # ------------------------------------------------
            # Sample action
            # ------------------------------------------------

            action = distribution.sample()

            # ------------------------------------------------
            # Save old log probability
            # ------------------------------------------------

            action_tensor = euclid.Tensor(
                np.array(
                    [action],
                    dtype=np.float32
                )
            )

            old_log_prob = distribution.log_prob(
                action_tensor
            )

            old_log_prob = float(
                np.asarray(
                    old_log_prob.data
                ).reshape(-1)[0]
            )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            next_state, reward, done = env.step(
                action
            )

            next_state_encoded = encode_state(
                next_state
            )

            # ------------------------------------------------
            # Store transition
            # ------------------------------------------------

            experience.add(
                Transition(
                    state=state_encoded,
                    action=action,
                    reward=reward,
                    next_state=next_state_encoded,
                    done=done,
                    log_prob=old_log_prob
                )
            )

            episode_reward += reward

            state = next_state
            steps += 1

        # ----------------------------------------------------
        # PPO update
        # ----------------------------------------------------

        if len(experience) > 0:

            agent.update(
                experience
            )

        episode_rewards.append(
            episode_reward
        )

        if (episode + 1) % 50 == 0:

            avg_reward = np.mean(
                episode_rewards[-50:]
            )

            print(
                f"  PPO Episode {episode + 1:3d} "
                f"- Avg Reward: {avg_reward:7.2f}"
            )

    return agent, env, episode_rewards


# ============================================================
# Q-Learning
# ============================================================

def train_qlearning(episodes=500):
    """Train Q-Learning agent."""

    env = CliffWalking()

    agent = QLearning(
        actions=lambda s: [0, 1, 2, 3],
        learning_rate=0.1,
        gamma=0.99
    )

    episode_rewards = []

    epsilon = 0.1

    for episode in range(episodes):

        state = env.reset()

        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:

            # Epsilon-greedy exploration
            if random.random() < epsilon:

                action = random.randint(
                    0,
                    3
                )

            else:

                q_values = [
                    agent.action_value(
                        state,
                        a
                    )
                    for a in range(4)
                ]

                action = q_values.index(
                    max(q_values)
                )

            next_state, reward, done = env.step(
                action
            )

            episode_reward += reward

            exp = Experience()

            exp.add(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
            )

            agent.update(
                exp
            )

            state = next_state
            steps += 1

        episode_rewards.append(
            episode_reward
        )

    return agent, env, episode_rewards


# ============================================================
# Moving Average
# ============================================================

def moving_average(
    values,
    window=50
):

    values = np.asarray(
        values
    )

    result = np.zeros_like(
        values,
        dtype=np.float64
    )

    for i in range(len(values)):

        start = max(
            0,
            i - window + 1
        )

        result[i] = np.mean(
            values[start:i + 1]
        )

    return result


# ============================================================
# Plot Comparison
# ============================================================

def plot_comparison(
    qlearning_rewards,
    ppo_rewards,
    save_path=None
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # --------------------------------------------------------
    # Training curves
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        qlearning_rewards,
        alpha=0.25,
        label="Q-Learning"
    )

    ax.plot(
        ppo_rewards,
        alpha=0.25,
        label="PPO"
    )

    window = 50

    qlearning_avg = moving_average(
        qlearning_rewards,
        window
    )

    ppo_avg = moving_average(
        ppo_rewards,
        window
    )

    ax.plot(
        qlearning_avg,
        linewidth=2.5,
        label="Q-Learning MA"
    )

    ax.plot(
        ppo_avg,
        linewidth=2.5,
        label="PPO MA"
    )

    ax.set_xlabel(
        "Episode",
        fontsize=12
    )

    ax.set_ylabel(
        "Total Reward",
        fontsize=12
    )

    ax.set_title(
        "Training Progress: Q-Learning vs PPO",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(
        fontsize=10
    )

    ax.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    ax = axes[1]

    algorithms = [
        "Q-Learning",
        "PPO"
    ]

    qlearning_final = np.mean(
        qlearning_rewards[-50:]
    )

    ppo_final = np.mean(
        ppo_rewards[-50:]
    )

    qlearning_best = max(
        qlearning_rewards
    )

    ppo_best = max(
        ppo_rewards
    )

    x = np.arange(
        len(algorithms)
    )

    width = 0.35

    final_bars = ax.bar(
        x - width / 2,
        [
            qlearning_final,
            ppo_final
        ],
        width,
        label="Final Avg (50 eps)",
        alpha=0.7
    )

    best_bars = ax.bar(
        x + width / 2,
        [
            qlearning_best,
            ppo_best
        ],
        width,
        label="Best Reward",
        alpha=0.7
    )

    ax.set_ylabel(
        "Reward",
        fontsize=12
    )

    ax.set_title(
        "Algorithm Comparison Summary",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        algorithms
    )

    ax.legend(
        fontsize=10
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y"
    )

    # --------------------------------------------------------
    # Bar labels
    # --------------------------------------------------------

    for bar in final_bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    for bar in best_bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    return fig


# ============================================================
# Print Comparison
# ============================================================

def print_comparison(
    qlearning_rewards,
    ppo_rewards
):

    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON: Q-LEARNING vs PPO")
    print("=" * 70)

    # --------------------------------------------------------
    # Q-Learning
    # --------------------------------------------------------

    print("\n" + "─" * 70)
    print("Q-LEARNING RESULTS")
    print("─" * 70)

    print(
        f"Initial 50-episode avg:    "
        f"{np.mean(qlearning_rewards[:50]):7.2f}"
    )

    print(
        f"Mid 250-episode avg:       "
        f"{np.mean(qlearning_rewards[200:250]):7.2f}"
    )

    print(
        f"Final 50-episode avg:      "
        f"{np.mean(qlearning_rewards[-50:]):7.2f}"
    )

    print(
        f"Best reward:               "
        f"{max(qlearning_rewards):7.2f}"
    )

    print(
        f"Worst reward:              "
        f"{min(qlearning_rewards):7.2f}"
    )

    print(
        f"Training stability:        "
        f"{np.std(qlearning_rewards[-50:]):7.2f} "
        f"(std dev)"
    )

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    print("\n" + "─" * 70)
    print("PPO RESULTS")
    print("─" * 70)

    print(
        f"Initial 50-episode avg:    "
        f"{np.mean(ppo_rewards[:50]):7.2f}"
    )

    print(
        f"Mid 250-episode avg:       "
        f"{np.mean(ppo_rewards[200:250]):7.2f}"
    )

    print(
        f"Final 50-episode avg:      "
        f"{np.mean(ppo_rewards[-50:]):7.2f}"
    )

    print(
        f"Best reward:               "
        f"{max(ppo_rewards):7.2f}"
    )

    print(
        f"Worst reward:              "
        f"{min(ppo_rewards):7.2f}"
    )

    print(
        f"Training stability:        "
        f"{np.std(ppo_rewards[-50:]):7.2f} "
        f"(std dev)"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n" + "─" * 70)
    print("KEY INSIGHTS")
    print("─" * 70)

    qlearning_final = np.mean(
        qlearning_rewards[-50:]
    )

    ppo_final = np.mean(
        ppo_rewards[-50:]
    )

    if qlearning_final > ppo_final:

        diff = (
            qlearning_final
            - ppo_final
        )

        print(
            f"✓ Q-Learning performs BETTER "
            f"by {diff:.2f} points"
        )

    elif ppo_final > qlearning_final:

        diff = (
            ppo_final
            - qlearning_final
        )

        print(
            f"✓ PPO performs BETTER "
            f"by {diff:.2f} points"
        )

    else:

        print(
            "✓ Both algorithms have "
            "the same final average reward"
        )

    print(
        "\n✓ Q-Learning uses a discrete Q-value table."
    )

    print(
        "✓ PPO uses neural-network policy and value functions."
    )

    print(
        "✓ PPO uses GAE and the clipped PPO objective."
    )

    print(
        "✓ Safe optimal behavior should approach "
        "the minimum number of non-cliff steps."
    )

    print(
        "✓ Cliff Walking is particularly well suited "
        "to comparing tabular and neural RL."
    )

    print("\n" + "=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("TRAINING COMPARISON: Q-LEARNING vs PPO")
    print("=" * 70)

    # --------------------------------------------------------
    # Q-Learning
    # --------------------------------------------------------

    print("\nTraining Q-Learning...")

    (
        qlearning_agent,
        qlearning_env,
        qlearning_rewards
    ) = train_qlearning(
        episodes=1000
    )

    print(
        f"✓ Q-Learning finished. "
        f"Final avg: "
        f"{np.mean(qlearning_rewards[-50:]):.2f}"
    )

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    print("\nTraining PPO with Euclid...")

    (
        ppo_agent,
        ppo_env,
        ppo_rewards
    ) = train_ppo(
        episodes=1000
    )

    print(
        f"✓ PPO finished. "
        f"Final avg: "
        f"{np.mean(ppo_rewards[-50:]):.2f}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print_comparison(
        qlearning_rewards,
        ppo_rewards
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    print("\nGenerating comparison plots...")

    viz_dir = (
        Path(__file__).parent
        / "visualizations"
    )

    viz_dir.mkdir(
        exist_ok=True
    )

    save_path = (
        viz_dir
        / "algorithm_comparison.png"
    )

    plot_comparison(
        qlearning_rewards,
        ppo_rewards,
        save_path=save_path
    )

    print(
        f"✓ Saved: {save_path}"
    )


if __name__ == "__main__":

    import matplotlib

    matplotlib.use("Agg")

    main()