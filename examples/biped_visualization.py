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


# ============================================================
# CONFIG
# ============================================================

STATE_DIM = 24
ACTION_DIM = 4
HIDDEN_DIM = 64

MAX_STEPS = 1600

SEED = 42

# Change this to:
#   actor_best.npz
#   actor_latest.npz
#   actor_final.npz
# ============================================================
# MODEL CHECKPOINT
# ============================================================

MODEL_DIR = (
    ROOT
    / "results"
    / "biped"
)


def find_latest_model():

    checkpoints = list(
        MODEL_DIR.glob("actor_*.npz")
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No actor checkpoints found in:\n{MODEL_DIR}"
        )

    return max(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
    )


MODEL_PATH = find_latest_model()

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

        self.log_std = euclid.Tensor(
            np.full(
                (1, ACTION_DIM),
                -0.75,
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
            euclid.Tensor,
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

        x = self.layer1.forward(state)

        x = euclid.layers.activations.Tanh().forward(x)

        x = self.layer2.forward(x)

        x = euclid.layers.activations.Tanh().forward(x)

        mean = self.mean_layer.forward(x)

        return mean

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    def parameters(self):

        return (
            self.layer1.parameters()
            + self.layer2.parameters()
            + self.mean_layer.parameters()
            + [self.log_std]
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model,
    path,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{path}"
        )

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
                    f"in checkpoint:\n{path}"
                )

            saved = checkpoint[key]

            if parameter.data.shape != saved.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"model={parameter.data.shape}, "
                    f"checkpoint={saved.shape}"
                )

            parameter.data[...] = saved


# ============================================================
# VISUALIZE
# ============================================================

def visualize():

    print("=" * 60)
    print("Archid PPO BipedalWalker Visualization")
    print("=" * 60)

    print(
        f"Loading actor:\n{MODEL_PATH}"
    )

    actor = Actor()

    load_model(
        actor,
        MODEL_PATH,
    )

    print("Actor loaded successfully.")
    print()

    env = gym.make(
        "BipedalWalker-v3",
        render_mode="human",
    )

    state, info = env.reset(
        seed=SEED
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

        # ----------------------------------------------------
        # Deterministic policy
        # ----------------------------------------------------
        #
        # The actor produces the Gaussian mean.
        # BipedalWalker expects [-1, 1], so apply tanh.
        #

        mean = actor.forward(
            state_tensor
        )

        mean = np.asarray(
            mean.data,
            dtype=np.float32,
        ).reshape(
            ACTION_DIM
        )

        action = np.tanh(
            mean
        ).astype(
            np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

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
        f"Reward: {total_reward:.2f}"
    )

    print(
        f"Steps:  {step + 1}"
    )

    env.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    visualize()