"""Reinforcement-learning algorithms."""

from .algorithm import Algorithm

from .action_value_based.action_based import ActionBased
from .action_value_based.qlearning import QLearning
from .action_value_based.sarsa import SARSA
from .action_value_based.dqn import DQN

from .state_value_based.state_based import StateBased
from .state_value_based.td import TD

from .actor_critic.actor_critic import ActorCritic
from .actor_critic.q_critic.q_critic import QCritic
from .actor_critic.q_critic.q_actor_critic import QActorCritic
from .actor_critic.v_critic.v_critic import VCritic
from .actor_critic.v_critic.ppo import PPO

__all__ = [
    "Algorithm",
    "ActionBased",
    "QLearning",
    "SARSA",
    "DQN",
    "StateBased",
    "TD",
    "ActorCritic",
    "QCritic",
    "QActorCritic",
    "VCritic",
    "PPO",
]
