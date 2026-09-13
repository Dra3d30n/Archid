# Archid AI Framework - Coding Instructions

## Project Overview
**Archid** is a high-level Python deep learning framework built on top of **Euclid ML**. It provides intuitive abstractions for reinforcement learning, supervised learning, and evolutionary learning algorithms, leveraging Euclid's lower-level tensor operations and optimization.

**Foundation:** Euclid ML (tensor computation, autodiff, optimization)
**Key Dependencies:** numpy, euclid-ml ≥ 0.9.9
**Build System:** setuptools with bdist_wheel
**Python Version:** 3.13+

### Relationship to Euclid ML
- **Euclid ML** is the computational backend (tensors, autodiff, optimizers)
- **Archid** provides algorithm abstractions and RL/SL/EL implementations on top of Euclid
- **Import Pattern:** Archid algorithms use `import euclid` for deep learning operations
- **Actor-Critic Methods:** PPO and Q-Actor-Critic rely on Euclid's neural network capabilities

## Architecture

### Core Structure
- **`archid/reinforcement_learning/`** - RL algorithms organized in hierarchies:
  - `algorithms/algorithm.py` - Base `Algorithm` class (abstract, defines `predict()` and `update()`)
  - `action_value_based/` - Action-value algorithms (Q-Learning, SARSA, DQN) inherit from `ActionBased`
  - `state_value_based/` - State-value algorithms (TD) inherit from `StateBased`
  - `actor_critic/` - Actor-critic variants (PPO, Q-Actor-Critic) inherit from `ActorCritic`
  
- **`archid/supervised_learning/`** - Supervised learning implementations
- **`archid/reinforcement_learning/environment/`** - Environment interface for RL
- **`archid/reinforcement_learning/experience/`** - Experience and Transition data classes

### Import Pattern
Public APIs are exposed through `__init__.py` files, starting from `archid/__init__.py` which imports from `reinforcement_learning/__init__.py` and `supervised_learning/__init__.py`.

## Critical Issues & Patterns

### Circular Import Bug (FIXED)
Previously, `state_value_based/state_based.py` had a self-import that's now fixed:
```python
# BEFORE (broken):
from .state_based import StateBased

# AFTER (fixed):
from ..algorithm import Algorithm
class StateBased(Algorithm):
    ...
```

### Euclid ML Integration
- **PPO and Actor-Critic algorithms** import euclid for neural network operations:
  ```python
  import euclid  # Used for actor/critic network forward passes
  ```
- **Value-based algorithms** (Q-Learning, SARSA, TD) are tabular and don't require Euclid
- **Planned expansions** would use Euclid for DQN and other deep RL methods
- **Current limitation:** Simplified PPO implementation uses mock networks (see examples)

### Inheritance Hierarchy
- All RL algorithms → `Algorithm` (base)
- Action-value algorithms → `ActionBased` → includes `td_target()`, `td_error()`, `action_value()`
- State-value algorithms → `StateBased` (similar interface, state-only)
- Policy gradient → `ActorCritic` with variants `QCritic`, `VCritic`

**Convention:** Each algorithm implements `update(experience)` which receives an `Experience` object containing `state`, `action`, `reward`, `next_state`, `next_action`, `done`.

## Development Workflow

### Build & Install
```bash
pip uninstall archid  # Clean previous
python setup.py bdist_wheel
pip install dist/archid-0.9.0-py3-none-any.whl
```

### Testing
```bash
python tests/test_reinforcement_learning.py  # Smoke tests for all RL algorithms
python tests/test_supervised_learning.py
```

**Important:** Tests import from installed package, not workspace directly. Always rebuild after code changes.

## Code Conventions

1. **Update Protocol:** Algorithms receive `Experience` objects with trajectory data; use `experience.state`, `experience.reward`, `experience.done`, etc.
2. **Table-Based Learning:** Tabular algorithms (Q-Learning, TD) store values in `self.v_table` or `self.q_table` dicts
3. **Discount Factor:** Use `self.gamma` (default 0.99) for future reward discounting
4. **Base Method Names:** `predict(state)`, `update(experience)`, `action_value()`, `state_value()`, `td_error()`, `td_target()`

## Key Files to Reference
- [archid/reinforcement_learning/algorithms/algorithm.py](../archid/reinforcement_learning/algorithms/algorithm.py) - Base algorithm interface
- [archid/reinforcement_learning/algorithms/action_value_based/action_based.py](../archid/reinforcement_learning/algorithms/action_value_based/action_based.py) - Action-value mixin (study for TD calculations)
- [archid/reinforcement_learning/experience/experience.py](../archid/reinforcement_learning/experience/experience.py) - Experience data structure
- [tests/test_reinforcement_learning.py](../tests/test_reinforcement_learning.py) - Algorithm usage patterns

## Common Pitfalls
- **Circular imports:** Check import paths in `__init__.py` files; each submodule should import upward/sideways, not self-reference
- **Missing builds:** Code changes aren't reflected in tests until `python setup.py bdist_wheel && pip install dist/*.whl`
- **Experience object access:** Always use dot notation (`experience.state`, not `experience['state']`)
- **Euclid integration:** Deep learning algorithms must properly import and use Euclid's tensor/autodiff capabilities; current simplified implementations use mock networks for examples
- **Neural network frameworks:** Actor-critic methods need integration with real deep learning (currently Euclid-based but simplified in examples)
