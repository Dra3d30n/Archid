import euclid as Euclid
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import archid
from archid.reinforcement_learning.reinforcement_learning import ReinforcementLearning
from euclid import PPO
from euclid import GaussianPolicy
from euclid import Episode
from euclid import Tensor

from euclid.backend import xp
# Network
network = Euclid.Sequential([
    Euclid.Dense(2, 32),
    Euclid.ReLU(),
    Euclid.Dense(32, 2),
])


# RL components
policy = GaussianPolicy(network)

optimizer = Euclid.Adam(
    learning_rate=0.001
)

algorithm = PPO(
    gamma=0.99
)


# Archid model
model = archid.model.Model(network,ReinforcementLearning(
    algorithm=algorithm,
    policy=policy,
    optimizer=optimizer
))


# Create fake episode
episode = Episode()

state = [0.0, 0.0]

for i in range(10):

    action = model.predict(Tensor(state))
    log_probability = model.learning.policy.log_probability(
        model.network.forward(Tensor(state)),
        action,
    )
    action=Euclid.to_xp(action).flatten()
    print(action.shape)
    next_state = [
        Euclid.to_xp(state[0]) + float(action[0]),
        Euclid.to_xp(state[1]) + float(action[1])
    ]

    reward = -(
        next_state[0] ** 2 +
        next_state[1] ** 2
    )

    done = i == 9

    episode.add(
        # Dense's backward pass expects a batch dimension, including for
        # one-sample PPO updates.
        state=Tensor([state]),
        action=action,
        reward=reward,
        next_state=Tensor(next_state),
        done=done,
        # PPO's old log probability is rollout data, not part of the current
        # optimization graph.
        log_probability=Tensor(log_probability.data)
    )

    state = next_state


# Train
print(model.learning.optimizer.parameters)
loss = model.learning.update_model(
    episode=episode,

)

print("Actor loss:", loss)
