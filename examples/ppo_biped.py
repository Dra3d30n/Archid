
"""
Archid PPO + Gymnasium BipedalWalker-v3
=======================================

Official Gymnasium BipedalWalker-v3 environment.

Actor:
    24 -> 64 Tanh -> 64 Tanh -> 4

Policy:
    Gaussian
    learned log_std
    tanh-squashed actions -> [-1, 1]

Critic:
    24 -> 64 Tanh -> 64 Tanh -> 1

Designed around Archid's existing PPO interface:

    distribution = actor.forward(state)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    experience.add(...)
    agent.update(experience)

"""

import sys
from pathlib import Path

import numpy as np
import euclid
import gymnasium as gym


# ============================================================
# ARCHID
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from archid.reinforcement_learning.algorithms.actor_critic.v_critic.ppo import PPO
from archid.reinforcement_learning.experience import (
    Experience,
    Transition,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MAX_STEPS = 1600

STATE_DIM = 24
ACTION_DIM = 4

HIDDEN_DIM = 64

# These are the values from the better-performing
# continuous-control version you were using earlier.
# ============================================================
# BIPEDALWALKER — GAIT / FORM FINE-TUNING
# Start from your current ~272 reward checkpoint
# ============================================================

ACTOR_LR = 7.5e-5       # lower than initial training
CRITIC_LR = 2.5e-4

GAMMA = 0.995           # slightly longer-term planning
GAE_LAMBDA = 0.97

CLIP_EPSILON = 0.12     # smaller policy updates
VALUE_CLIP_EPSILON = 0.15

VALUE_COEF = 0.5
ENTROPY_COEF = 0.005    # retain a little exploration

PPO_EPOCHS = 10
MINIBATCH_SIZE = 256

MAX_GRAD_NORM = 0.5
LOG_STD_INIT = -1.0
TARGET_KL = 0.015

# Continue with the existing policy's exploration level.
# Don't reset log_std when loading the checkpoint.
# ============================================================
# LONG-RUN TRAINING
# ============================================================

EPISODES = 2000
RESUME = True
# Save every N episodes.
CHECKPOINT_EVERY = 100

# Directory for all BipedalWalker results.
RESULTS_DIR = ROOT / "results" / "biped"

ACTOR_LATEST_PATH = RESULTS_DIR / "actor_latest.npz"
CRITIC_LATEST_PATH = RESULTS_DIR / "critic_latest.npz"

ACTOR_BEST_PATH = RESULTS_DIR / "actor_best.npz"
CRITIC_BEST_PATH = RESULTS_DIR / "critic_best.npz"

ACTOR_FINAL_PATH = RESULTS_DIR / "actor_final.npz"
CRITIC_FINAL_PATH = RESULTS_DIR / "critic_final.npz"

HISTORY_PATH = RESULTS_DIR / "history.npy"
PROGRESS_PATH = RESULTS_DIR / "progress.npz"
# ============================================================
# GAUSSIAN DISTRIBUTION
# ============================================================
class GaussianDistribution:

    def __init__(self, mean, log_std):
        self.mean = mean
        self.log_std = log_std

    def _log_std(self):
        return np.clip(
            self.log_std,
            -5.0,
            0.0,
        )

    def sample(self):

        mean = np.asarray(
            self.mean.data,
            dtype=np.float32,
        ).reshape(-1)

        log_std = np.asarray(
            self.log_std.data,
            dtype=np.float32,
        ).reshape(-1)

        log_std = np.clip(
            log_std,
            -5.0,
            0.0,
        )

        std = np.exp(log_std)

        noise = np.random.randn(
            ACTION_DIM
        ).astype(np.float32)

        z = mean + std * noise

        action = np.tanh(z)

        return action.astype(np.float32)

    def log_prob(self, actions):

        if hasattr(actions, "data"):
            actions = actions.data

        actions = np.asarray(
            actions,
            dtype=np.float32,
        )

        if actions.ndim == 1:
            actions = actions.reshape(
                1,
                ACTION_DIM,
            )

        elif (
            actions.ndim == 3
            and actions.shape[1] == 1
        ):
            actions = actions[:, 0, :]

        if actions.ndim != 2:
            raise ValueError(
                f"Expected actions (N, {ACTION_DIM}), "
                f"got {actions.shape}"
            )

        if actions.shape[1] != ACTION_DIM:
            raise ValueError(
                f"Expected actions (N, {ACTION_DIM}), "
                f"got {actions.shape}"
            )

        # Recover the pre-tanh Gaussian variable.
        clipped = np.clip(
            actions,
            -0.999999,
            0.999999,
        )

        z = np.arctanh(
            clipped
        ).astype(
            np.float32
        )

        z = euclid.Tensor(z)

        log_std = euclid.Tensor(
            np.clip(
                np.asarray(
                    self.log_std.data,
                    dtype=np.float32,
                ),
                -5.0,
                0.0,
            )
        )

        std = log_std.exp()

        difference = (
            z - self.mean
        )

        variance = (
            std * std
        )

        gaussian_log_prob = -0.5 * (
            (
                difference
                * difference
            )
            / (
                variance + 1e-8
            )
            + 2.0 * log_std
            + np.log(
                2.0 * np.pi
            )
        )

        # tanh change-of-variables correction.
        action_tensor = euclid.Tensor(
            clipped
        )

        correction = (
            1.0
            - action_tensor * action_tensor
            + 1e-6
        ).log()

        return (
            gaussian_log_prob
            - correction
        ).sum(
            axis=-1
        )

    def entropy(self):

        log_std = euclid.Tensor(
            np.clip(
                np.asarray(
                    self.log_std.data,
                    dtype=np.float32,
                ),
                -5.0,
                0.0,
            )
        )

        entropy = (
            log_std
            + 0.5 * np.log(
                2.0 * np.pi * np.e
            )
        )

        return entropy.sum(
            axis=-1
        )


# ============================================================
# ACTOR
# ============================================================

class Actor:

    def __init__(self):

        self.layer1 = euclid.layers.Dense(
            STATE_DIM,
            HIDDEN_DIM,
        )

        self.layer2 = euclid.layers.Dense(
            HIDDEN_DIM,
            HIDDEN_DIM,
        )

        self.mean_layer = euclid.layers.Dense(
            HIDDEN_DIM,
            ACTION_DIM,
        )

        # One learned std per action.
        #
        # Shape:
        #     (1, 4)
        #
        self.log_std = euclid.Tensor(
            np.full(
                (1, ACTION_DIM),
                LOG_STD_INIT,
                dtype=np.float32,
            ),
            requires_grad=True,
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, state):

        if not isinstance(
            state,
            euclid.Tensor
        ):

            state = euclid.Tensor(
                np.asarray(
                    state,
                    dtype=np.float32,
                ).reshape(
                    1,
                    STATE_DIM,
                )
            )

        x = self.layer1.forward(
            state
        )

        x = euclid.layers.activations.Tanh().forward(
            x
        )

        x = self.layer2.forward(
            x
        )

        x = euclid.layers.activations.Tanh().forward(
            x
        )

        mean = self.mean_layer.forward(
            x
        )

        # IMPORTANT:
        #
        # The mean itself is NOT tanh'd.
        #
        # Gaussian noise is added first and the final
        # sampled action is squashed in GaussianDistribution.
        #
        return GaussianDistribution(
            mean,
            self.log_std,
        )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    def parameters(self):

        return (
            self.layer1.parameters()
            + self.layer2.parameters()
            + self.mean_layer.parameters()
            + [
                self.log_std
            ]
        )

    def __call__(self, state):

        return self.forward(state)


# ============================================================
# CRITIC
# ============================================================

class Critic:

    def __init__(self):

        self.layer1 = euclid.layers.Dense(
            STATE_DIM,
            HIDDEN_DIM,
        )

        self.layer2 = euclid.layers.Dense(
            HIDDEN_DIM,
            HIDDEN_DIM,
        )

        self.value_layer = euclid.layers.Dense(
            HIDDEN_DIM,
            1,
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, state):

        if not isinstance(
            state,
            euclid.Tensor
        ):

            state = euclid.Tensor(
                np.asarray(
                    state,
                    dtype=np.float32,
                ).reshape(
                    1,
                    STATE_DIM,
                )
            )

        x = self.layer1.forward(
            state
        )

        x = euclid.layers.activations.Tanh().forward(
            x
        )

        x = self.layer2.forward(
            x
        )

        x = euclid.layers.activations.Tanh().forward(
            x
        )

        return self.value_layer.forward(
            x
        )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    def parameters(self):

        return (
            self.layer1.parameters()
            + self.layer2.parameters()
            + self.value_layer.parameters()
        )

    def __call__(self, state):

        return self.forward(state)

# ============================================================
# MODEL CHECKPOINTS
# ============================================================

def save_model(
    model,
    path,
):
    """
    Save all trainable parameters of a model.

    The model architecture is recreated by Actor()/Critic(),
    while this file stores the actual learned parameter arrays.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters = model.parameters()

    arrays = {}

    for index, parameter in enumerate(parameters):

        arrays[f"param_{index}"] = np.asarray(
            parameter.data,
            dtype=np.float32,
        ).copy()

    np.savez_compressed(
        path,
        **arrays,
    )


def load_model(
    model,
    path,
):
    """
    Load saved parameters into an already-created model.
    """

    with np.load(
        path,
        allow_pickle=False,
    ) as checkpoint:

        parameters = model.parameters()

        for index, parameter in enumerate(parameters):

            key = f"param_{index}"

            if key not in checkpoint:
                raise ValueError(
                    f"Missing parameter '{key}' "
                    f"in checkpoint: {path}"
                )

            parameter.data[...] = checkpoint[key]

def load_training_checkpoint(agent):
    """
    Load the latest actor, critic, reward history,
    and training progress.

    Returns:
        history
        best_average
        next_episode
    """

    actor_exists = ACTOR_LATEST_PATH.exists()
    critic_exists = CRITIC_LATEST_PATH.exists()
    progress_exists = PROGRESS_PATH.exists()
    history_exists = HISTORY_PATH.exists()

    # --------------------------------------------------------
    # No checkpoint
    # --------------------------------------------------------

    if not (
        actor_exists
        and critic_exists
        and progress_exists
        and history_exists
    ):
        print()
        print(
            "[RESUME] No complete checkpoint found."
        )
        print(
            "[RESUME] Starting from episode 1."
        )
        print()

        return (
            [],
            -np.inf,
            1,
        )

    # --------------------------------------------------------
    # Load actor
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING TRAINING CHECKPOINT")
    print("=" * 60)

    print(
        f"Actor : {ACTOR_LATEST_PATH}"
    )

    load_model(
        agent.actor,
        ACTOR_LATEST_PATH,
    )

    # --------------------------------------------------------
    # Load critic
    # --------------------------------------------------------

    print(
        f"Critic: {CRITIC_LATEST_PATH}"
    )

    load_model(
        agent.critic,
        CRITIC_LATEST_PATH,
    )

    # --------------------------------------------------------
    # Load history
    # --------------------------------------------------------

    history = np.load(
        HISTORY_PATH
    ).astype(
        np.float32
    ).tolist()

    # --------------------------------------------------------
    # Load progress
    # --------------------------------------------------------

    with np.load(
        PROGRESS_PATH,
        allow_pickle=False,
    ) as progress:

        saved_episode = int(
            progress["episode"]
        )

        best_average = float(
            progress["best_average"]
        )

    next_episode = saved_episode + 1

    print(
        f"Saved episode:   {saved_episode:,}"
    )

    print(
        f"Next episode:    {next_episode:,}"
    )

    print(
        f"Best Avg100:     {best_average:.2f}"
    )

    print(
        f"History entries: {len(history):,}"
    )

    print("=" * 60)
    print()

    return (
        history,
        best_average,
        next_episode,
    )
def save_checkpoint(
    agent,
    history,
    episode,
    best_average,
):
    """
    Save the current actor and critic plus training progress.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Latest models
    # --------------------------------------------------------

    save_model(
        agent.actor,
        ACTOR_LATEST_PATH,
    )

    save_model(
        agent.critic,
        CRITIC_LATEST_PATH,
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    np.save(
        HISTORY_PATH,
        np.asarray(
            history,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------------
    # Training metadata
    # --------------------------------------------------------

    np.savez(
        PROGRESS_PATH,
        episode=np.asarray(
            episode,
            dtype=np.int64,
        ),
        best_average=np.asarray(
            best_average,
            dtype=np.float32,
        ),
    )

    print()
    print(
        f"[CHECKPOINT] Episode {episode}"
    )

    print(
        f"  Actor : {ACTOR_LATEST_PATH}"
    )

    print(
        f"  Critic: {CRITIC_LATEST_PATH}"
    )

    print()


def save_best_models(
    agent,
):
    """
    Save copies of the best-performing actor and critic.
    """

    save_model(
        agent.actor,
        ACTOR_BEST_PATH,
    )

    save_model(
        agent.critic,
        CRITIC_BEST_PATH,
    )
# ============================================================
# PPO AGENT
# ============================================================

def create_agent():

    actor = Actor()

    critic = Critic()

    # --------------------------------------------------------
    # Actor optimizer
    # --------------------------------------------------------

    actor_optimizer = euclid.optimizers.Adam(
        learning_rate=ACTOR_LR
    )

    actor_optimizer.set_parameters(
        actor.parameters()
    )

    # --------------------------------------------------------
    # Critic optimizer
    # --------------------------------------------------------

    critic_optimizer = euclid.optimizers.Adam(
        learning_rate=CRITIC_LR
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

        epochs=PPO_EPOCHS,
    )

    return agent


# ============================================================
# TRAIN
# ============================================================
# ============================================================
# TRAIN
# ============================================================
def train():

    np.random.seed(
        SEED
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = gym.make(
        "BipedalWalker-v3"
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = create_agent()

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if RESUME:

        (
            history,
            best_average,
            start_episode,
        ) = load_training_checkpoint(
            agent
        )

    else:

        history = []

        best_average = -np.inf

        start_episode = 1

        print()
        print(
            "[RESUME] Disabled."
        )
        print(
            "[RESUME] Starting from episode 1."
        )
        print()

    # --------------------------------------------------------
    # Already finished?
    # --------------------------------------------------------

    if start_episode > EPISODES:

        print(
            f"Training already reached "
            f"episode {start_episode - 1:,}."
        )

        env.close()

        return agent, history

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for episode in range(
        start_episode,
        EPISODES + 1,
    ):

        state, info = env.reset(
            seed=SEED + episode
        )

        experience = Experience()

        total_reward = 0.0

        # ----------------------------------------------------
        # Rollout
        # ----------------------------------------------------

        for step in range(
            MAX_STEPS
        ):

            state_array = np.asarray(
                state,
                dtype=np.float32,
            )

            state_tensor = euclid.Tensor(
                state_array.reshape(
                    1,
                    STATE_DIM,
                )
            )

            # ------------------------------------------------
            # Policy
            # ------------------------------------------------

            distribution = (
                agent.actor.forward(
                    state_tensor
                )
            )

            # ------------------------------------------------
            # Sample action
            # ------------------------------------------------

            action = distribution.sample()

            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(
                ACTION_DIM
            )

            action = np.clip(
                action,
                -1.0,
                1.0,
            )

            # ------------------------------------------------
            # Log probability
            # ------------------------------------------------

            log_prob = (
                distribution.log_prob(
                    action
                )
            )

            log_prob_value = float(
                np.asarray(
                    log_prob.data
                ).reshape(-1)[0]
            )

            # ------------------------------------------------
            # Environment
            # ------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            done = (
                terminated
                or truncated
            )

            total_reward += float(
                reward
            )

            # ------------------------------------------------
            # Experience
            # ------------------------------------------------

            experience.add(
                Transition(
                    state=state_array.copy(),

                    action=action.copy(),

                    reward=float(
                        reward
                    ),

                    next_state=np.asarray(
                        next_state,
                        dtype=np.float32,
                    ).copy(),

                    done=done,

                    log_prob=log_prob_value,
                )
            )

            state = next_state

            if done:
                break

        # ----------------------------------------------------
        # PPO update
        # ----------------------------------------------------

        losses = None

        if len(experience) > 0:

            losses = agent.update(
                experience
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            total_reward
        )

        recent = history[-100:]

        average_reward = float(
            np.mean(
                recent
            )
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if episode % 10 == 0:

            print(
                f"Episode {episode:5d} | "
                f"Reward {total_reward:9.2f} | "
                f"Avg100 {average_reward:9.2f} | "
                f"Steps {len(experience):4d}"
            )

            if losses is not None:
                print(
                    f"                 Losses: {losses}"
                )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (
            len(history) >= 100
            and average_reward > best_average
        ):

            best_average = average_reward

            save_best_models(
                agent
            )

            # Don't spam the terminal every episode.
            if episode % 10 == 0:

                print(
                    f"[BEST] Episode {episode} | "
                    f"Avg100 {best_average:.2f}"
                )

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if episode % CHECKPOINT_EVERY == 0:

            save_checkpoint(
                agent=agent,
                history=history,
                episode=episode,
                best_average=best_average,
            )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    save_model(
        agent.actor,
        ACTOR_FINAL_PATH,
    )

    save_model(
        agent.critic,
        CRITIC_FINAL_PATH,
    )

    np.save(
        HISTORY_PATH,
        np.asarray(
            history,
            dtype=np.float32,
        ),
    )

    np.savez(
        PROGRESS_PATH,
        episode=np.asarray(
            EPISODES,
            dtype=np.int64,
        ),
        best_average=np.asarray(
            best_average,
            dtype=np.float32,
        ),
    )

    print()
    print("=" * 60)
    print("FINAL MODELS SAVED")
    print("=" * 60)

    print(
        f"Actor       : {ACTOR_FINAL_PATH}"
    )

    print(
        f"Critic      : {CRITIC_FINAL_PATH}"
    )

    print(
        f"Best actor  : {ACTOR_BEST_PATH}"
    )

    print(
        f"Best critic : {CRITIC_BEST_PATH}"
    )

    env.close()

    return agent, history
# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    agent,
    episodes=1,
):

    env = gym.make(
        "BipedalWalker-v3",
        render_mode="human",
    )

    print()
    print("=" * 60)
    print("Visual evaluation")
    print("=" * 60)

    for episode in range(
        episodes
    ):

        state, info = env.reset(
            seed=SEED + 10000 + episode
        )

        total_reward = 0.0

        for step in range(
            MAX_STEPS
        ):

            state_tensor = euclid.Tensor(
                np.asarray(
                    state,
                    dtype=np.float32,
                ).reshape(
                    1,
                    STATE_DIM,
                )
            )

            distribution = (
                agent.actor.forward(
                    state_tensor
                )
            )

            # ------------------------------------------------
            # Deterministic evaluation
            # ------------------------------------------------
            #
            # Mean is the deterministic Gaussian action.
            # The environment action is tanh(mean).
            #

            mean = np.asarray(
                distribution.mean.data,
                dtype=np.float32,
            ).reshape(
                ACTION_DIM
            )

            action = np.tanh(
                mean
            ).astype(
                np.float32
            )

            (
                state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            total_reward += float(
                reward
            )

            if (
                terminated
                or truncated
            ):
                break

        print(
            f"Episode {episode + 1}: "
            f"{total_reward:.2f} "
            f"({step + 1} steps)"
        )

    env.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print(
        "Archid PPO + Gymnasium "
        "BipedalWalker-v3"
    )
    print("=" * 60)

    print(
        f"Observation dimension: {STATE_DIM}"
    )

    print(
        f"Action dimension:      {ACTION_DIM}"
    )

    print(
        f"Training episodes:     {EPISODES:,}"
    )

    print(
        f"Maximum steps:         {MAX_STEPS}"
    )

    print(
        f"Actor learning rate:   {ACTOR_LR}"
    )

    print(
        f"Critic learning rate:  {CRITIC_LR}"
    )

    print(
        f"Initial log std:       {LOG_STD_INIT}"
    )

    print(
        f"PPO epochs:            {PPO_EPOCHS}"
    )

    print(
        f"Checkpoint interval:   {CHECKPOINT_EVERY}"
    )

    print(
        f"Results directory:     {RESULTS_DIR}"
    )

    print("=" * 60)
    print()

    agent, history = train()

    evaluate(
        agent,
        episodes=1,
    )

    print()
    print(
        "Training and evaluation complete."
    )