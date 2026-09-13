
import numpy as np
import euclid
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from archid.reinforcement_learning.algorithms.actor_critic.v_critic.ppo import PPO
from archid.reinforcement_learning.environment import Environment
from archid.reinforcement_learning.experience import Experience, Transition


# ============================================================
# CATEGORICAL DISTRIBUTION
# ============================================================
class CategoricalDistribution:

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def sample(self):

        probs = np.asarray(
            self.probabilities.data,
            dtype=np.float64
        ).reshape(-1)

        probs = np.maximum(
            probs,
            1e-8
        )

        probs /= probs.sum()

        return int(
            np.random.choice(
                len(probs),
                p=probs
            )
        )
    def log_prob(self, actions):

        probabilities = self.probabilities

        # ========================================================
        # Single action
        # ========================================================

        if np.isscalar(actions):

            action = int(actions)

            probability = probabilities[
                0,
                action
            ]

            return (
                probability + 1e-8
            ).log()

        # ========================================================
        # Batch actions
        # ========================================================

        if hasattr(actions, "data"):
            actions = actions.data

        actions = np.asarray(
            actions,
            dtype=np.int64
        ).reshape(-1, 1)

        selected = probabilities.gather(
            axis=1,
            indices=actions
        )

        selected = selected.reshape(-1)

        return (
            selected + 1e-8
        ).log()
    def entropy(self):
        probabilities = self.probabilities

        return -(
            probabilities *
            (probabilities + 1e-8).log()
        ).sum(axis=-1)
# ============================================================
# CONFIG
# ============================================================

EPISODES = 2000
MAX_STEPS = 1000

STATE_DIM = 10
ACTION_DIM = 9

DT = 0.05

GRAVITY = 9.81

LEG_MASS = 1.0
LEG_LENGTH = 1.0
HIP_HEIGHT = 1.2

TORQUE = 2.5
FRICTION = 0.15

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2

ACTOR_LR = 0.0001
CRITIC_LR = 0.0003

PPO_EPOCHS = 10


# ============================================================
# ACTION TABLE
# ============================================================

ACTION_TABLE = []

for left in [-1, 0, 1]:

    for right in [-1, 0, 1]:

        ACTION_TABLE.append(
            (left, right)
        )


# ============================================================
# BIPED ENVIRONMENT
# ============================================================

class BipedalWalking(Environment):

    def __init__(self):

        self.reset()

    def reset(self):

        self.x = 0.0
        self.x_velocity = 0.0

        self.body_angle = 0.0
        self.body_angular_velocity = 0.0

        self.left_angle = 0.20
        self.right_angle = -0.20

        self.left_angular_velocity = 0.0
        self.right_angular_velocity = 0.0

        self.steps = 0

        return self._state()

    def _state(self):

        state = np.array([

            self.x_velocity,

            self.body_angle,
            self.body_angular_velocity,

            self.left_angle,
            self.left_angular_velocity,

            self.right_angle,
            self.right_angular_velocity,

            np.sin(self.left_angle),
            np.sin(self.right_angle),

            np.cos(self.body_angle),

        ], dtype=np.float32)

        return euclid.Tensor(state)

    def step(self, action):

        left_torque, right_torque = ACTION_TABLE[
            int(action)
        ]

        left_torque *= TORQUE
        right_torque *= TORQUE

        # ----------------------------------------------------
        # Leg physics
        # ----------------------------------------------------

        left_gravity = (
            -GRAVITY
            * np.sin(self.left_angle)
            / LEG_LENGTH
        )

        right_gravity = (
            -GRAVITY
            * np.sin(self.right_angle)
            / LEG_LENGTH
        )

        self.left_angular_velocity += (
            left_torque
            + left_gravity
            - FRICTION
            * self.left_angular_velocity
        ) * DT

        self.right_angular_velocity += (
            right_torque
            + right_gravity
            - FRICTION
            * self.right_angular_velocity
        ) * DT

        self.left_angle += (
            self.left_angular_velocity
            * DT
        )

        self.right_angle += (
            self.right_angular_velocity
            * DT
        )

        # ----------------------------------------------------
        # Body dynamics
        # ----------------------------------------------------

        leg_difference = (
            self.left_angle
            - self.right_angle
        )

        body_torque = (
            -0.8 * self.body_angle
            + 0.35 * leg_difference
            + 0.08 * (
                left_torque
                - right_torque
            )
        )

        self.body_angular_velocity += (
            body_torque * DT
        )

        self.body_angular_velocity *= 0.98

        self.body_angle += (
            self.body_angular_velocity
            * DT
        )

        # ----------------------------------------------------
        # Forward movement
        # ----------------------------------------------------

        leg_drive = (
            np.sin(self.left_angle)
            - np.sin(self.right_angle)
        )

        self.x_velocity += (
            0.12 * leg_drive
            - 0.08 * self.x_velocity
        ) * DT

        self.x += (
            self.x_velocity * DT
        )

        # ----------------------------------------------------
        # Termination
        # ----------------------------------------------------

        self.steps += 1

        fallen = (
            abs(self.body_angle) > 0.9
            or abs(self.left_angle) > 1.5
            or abs(self.right_angle) > 1.5
        )

        done = (
            fallen
            or self.steps >= MAX_STEPS
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        forward_reward = self.x_velocity

        upright_reward = (
            1.0
            - abs(self.body_angle)
        )

        alternating_reward = (
            -abs(
                self.left_angle
                + self.right_angle
            )
        )

        energy_penalty = (
            0.002
            * (
                abs(left_torque)
                + abs(right_torque)
            )
        )

        reward = (
            2.0 * forward_reward
            + 0.3 * upright_reward
            + 0.15 * alternating_reward
            - energy_penalty
        )

        if fallen:

            reward -= 10.0

        return (
            self._state(),
            float(reward),
            done
        )

    def render(self):

        print(
            f"x={self.x:7.3f} "
            f"velocity={self.x_velocity:7.3f} "
            f"body={self.body_angle:7.3f} "
            f"L={self.left_angle:7.3f} "
            f"R={self.right_angle:7.3f}"
        )

    def close(self):
        pass


# ============================================================
# ACTOR
# ============================================================

def make_actor():

    return euclid.Sequential([
        euclid.Dense(
            STATE_DIM,
            64
        ),

        euclid.Tanh(),

        euclid.Dense(
            64,
            64
        ),

        euclid.Tanh(),

        euclid.Dense(
            64,
            ACTION_DIM
        ),

        euclid.Softmax()
    ])


# ============================================================
# DISTRIBUTIONAL ACTOR
# ============================================================

class CategoricalActor:

    def __init__(self, network):

        self.network = network

    def forward(self, state):

        probabilities = self.network.forward(
            state
        )

        return CategoricalDistribution(
            probabilities
        )

    def parameters(self):

        return self.network.parameters()


# ============================================================
# CRITIC
# ============================================================

def make_critic():

    return euclid.Sequential([
        euclid.Dense(
            STATE_DIM,
            64
        ),

        euclid.Tanh(),

        euclid.Dense(
            64,
            64
        ),

        euclid.Tanh(),

        euclid.Dense(
            64,
            1
        )
    ])


# ============================================================
# OPTIMIZERS
# ============================================================

def make_optimizer(parameters, lr):

    optimizer = euclid.Adam(lr)

    optimizer.set_parameters(
        parameters
    )

    return optimizer


# ============================================================
# TRAINING
# ============================================================

def train():

    env = BipedalWalking()

    actor_network = make_actor()

    actor = CategoricalActor(
        actor_network
    )

    critic = make_critic()

    actor_optimizer = make_optimizer(
        actor.parameters(),
        ACTOR_LR
    )

    critic_optimizer = make_optimizer(
        critic.parameters(),
        CRITIC_LR
    )

    agent = PPO(
        actor=actor,
        critic=critic,

        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,

        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,

        clip_epsilon=CLIP_EPSILON,

        value_coef=0.5,
        entropy_coef=0.01,

        epochs=PPO_EPOCHS
    )

    reward_history = []

    for episode in range(EPISODES):

        state = env.reset()

        experience = Experience()

        total_reward = 0.0

        for step in range(MAX_STEPS):

            # ------------------------------------------------
            # Actor
            # ------------------------------------------------

            distribution = agent.actor.forward(
                state
            )

            action = distribution.sample()

            log_prob = distribution.log_prob(
                action
            )

            log_prob_value = float(
                np.asarray(
                    log_prob.data
                ).reshape(-1)[0]
            )

            # ------------------------------------------------
            # Environment
            # ------------------------------------------------

            next_state, reward, done = env.step(
                action
            )

            total_reward += reward

            # ------------------------------------------------
            # Store transition
            # ------------------------------------------------

            experience.add(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob_value
                )
            )

            state = next_state

            if done:

                break

        # ----------------------------------------------------
        # PPO update
        # ----------------------------------------------------

        losses = agent.update(
            experience
        )

        reward_history.append(
            total_reward
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if episode % 10 == 0:

            recent = reward_history[-10:]

            print(
                f"Episode {episode:4d} | "
                f"Reward {total_reward:8.2f} | "
                f"Avg {np.mean(recent):8.2f} | "
                f"Steps {step + 1:3d} | "
                f"X {env.x:7.3f}"
            )

    return agent, reward_history


# ============================================================
# TEST
# ============================================================
def test(agent):

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    env = BipedalWalking()

    state = env.reset()

    frames = []

    # --------------------------------------------------------
    # Run the final policy and record the trajectory
    # --------------------------------------------------------

    total_reward = 0.0

    for step in range(MAX_STEPS):

        distribution = agent.actor.forward(state)

        probabilities = (
            distribution.probabilities.data
            .reshape(-1)
        )

        # Deterministic policy
        action = int(
            np.argmax(probabilities)
        )

        # Save state BEFORE stepping
        frames.append({
            "x": env.x,
            "body_angle": env.body_angle,
            "left_angle": env.left_angle,
            "right_angle": env.right_angle,
        })

        state, reward, done = env.step(action)

        total_reward += reward

        if done:
            break

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("================================")
    print("TEST")
    print("================================")

    print(
        f"Steps: {step + 1}"
    )

    print(
        f"Distance: {env.x:.3f}"
    )

    print(
        f"Reward: {total_reward:.3f}"
    )

    print(
        f"Fallen: "
        f"{done and step + 1 < MAX_STEPS}"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.set_aspect("equal")

    ax.set_ylim(
        -0.2,
        2.0
    )

    ax.set_xlim(
        -1.0,
        5.0
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Height")

    ax.set_title(
        "Archid PPO Bipedal Walking"
    )

    # Ground
    ax.axhline(
        0.0,
        linewidth=2
    )

    body_line, = ax.plot(
        [],
        [],
        linewidth=5
    )

    left_leg_line, = ax.plot(
        [],
        [],
        linewidth=4
    )

    right_leg_line, = ax.plot(
        [],
        [],
        linewidth=4
    )

    foot_points, = ax.plot(
        [],
        [],
        "o",
        markersize=7
    )

    # --------------------------------------------------------
    # Coordinate calculation
    # --------------------------------------------------------

    def get_coordinates(frame):

        x = frame["x"]
        body_angle = frame["body_angle"]

        left_angle = frame["left_angle"]
        right_angle = frame["right_angle"]

        hip_height = HIP_HEIGHT

        # Body
        body_length = 0.65

        body_dx = (
            np.sin(body_angle)
            * body_length
            / 2.0
        )

        body_dy = (
            np.cos(body_angle)
            * body_length
            / 2.0
        )

        body_x = [
            x - body_dx,
            x + body_dx
        ]

        body_y = [
            hip_height - body_dy,
            hip_height + body_dy
        ]

        # Hip
        hip_x = x
        hip_y = hip_height

        # Legs
        left_foot_x = (
            hip_x
            + np.sin(left_angle)
            * LEG_LENGTH
        )

        left_foot_y = (
            hip_y
            - np.cos(left_angle)
            * LEG_LENGTH
        )

        right_foot_x = (
            hip_x
            + np.sin(right_angle)
            * LEG_LENGTH
        )

        right_foot_y = (
            hip_y
            - np.cos(right_angle)
            * LEG_LENGTH
        )

        return (
            body_x,
            body_y,
            [hip_x, left_foot_x],
            [hip_y, left_foot_y],
            [hip_x, right_foot_x],
            [hip_y, right_foot_y],
            [left_foot_x, right_foot_x],
            [left_foot_y, right_foot_y],
        )

    # --------------------------------------------------------
    # Animation
    # --------------------------------------------------------

    def update(frame_index):

        frame = frames[frame_index]

        (
            body_x,
            body_y,
            left_x,
            left_y,
            right_x,
            right_y,
            feet_x,
            feet_y,
        ) = get_coordinates(frame)

        body_line.set_data(
            body_x,
            body_y
        )

        left_leg_line.set_data(
            left_x,
            left_y
        )

        right_leg_line.set_data(
            right_x,
            right_y
        )

        foot_points.set_data(
            feet_x,
            feet_y
        )

        # Follow the agent
        center_x = frame["x"]

        ax.set_xlim(
            center_x - 3.0,
            center_x + 3.0
        )

        return (
            body_line,
            left_leg_line,
            right_leg_line,
            foot_points,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=50,
        blit=True
    )

    plt.show()
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    agent, history = train()

    test(agent)
