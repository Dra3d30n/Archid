# Euclid ML Integration in Archid

## Overview
Archid is built on top of **Euclid ML**, a lower-level deep learning framework providing:
- **Tensor operations** - Efficient array computations
- **Automatic differentiation** - Gradient computation for backprop
- **Optimization algorithms** - SGD, Adam, and other optimizers
- **Neural network layers** - Dense, Conv, RNN primitives

## Architecture Relationship

```
┌─────────────────────────────────────────┐
│  Archid (High-Level RL/SL Framework)    │
│  - RL algorithms (PPO, DQN, Q-Learning) │
│  - SL models (supervised learning)      │
│  - Environment abstractions             │
└──────────────────┬──────────────────────┘
                   │ imports euclid
┌──────────────────▼──────────────────────┐
│   Euclid ML (Low-Level Math Engine)     │
│   - Tensor computation                  │
│   - Automatic differentiation           │
│   - Optimizers (SGD, Adam, etc)         │
│   - Neural network building blocks      │
└─────────────────────────────────────────┘
```

## Current Integration Status

### Algorithms Using Euclid ✅
1. **PPO (Proximal Policy Optimization)**
   - File: `archid/reinforcement_learning/algorithms/actor_critic/v_critic/ppo.py`
   - Uses: `import euclid` for actor/critic networks
   - Status: Framework in place, needs proper network implementation

2. **Q-Actor-Critic**
   - File: `archid/reinforcement_learning/algorithms/actor_critic/q_critic/q_actor_critic.py`
   - Uses: `import euclid` for value function networks
   - Status: Framework in place

### Tabular Algorithms (No Euclid) ✅
- **Q-Learning** - Uses dictionary-based Q-table
- **SARSA** - Uses dictionary-based state-action values
- **TD (Temporal Difference)** - Uses dictionary-based state values
- Status: Fully functional, no neural networks needed

### Algorithms Needing Euclid Integration
- **DQN** - Would use Euclid for deep Q-networks
- **A2C/A3C** - Would use Euclid for neural policy/value networks
- **DDPG** - Would use Euclid for continuous control networks

## How to Use Euclid in Archid

### Example: Creating a Simple Neural Network
```python
import euclid
import numpy as np

# Create a neural network layer
layer = euclid.Dense(input_dim=48, output_dim=64, activation='relu')

# Forward pass
state = np.zeros(48)
output = layer.forward(state)

# Get gradients
gradients = layer.gradients()
```

### Example: Policy Network in Actor-Critic
```python
import euclid

class ActorNetwork:
    def __init__(self, state_dim=48, action_dim=4):
        self.layer1 = euclid.Dense(state_dim, 64, activation='relu')
        self.layer2 = euclid.Dense(64, action_dim, activation='softmax')
    
    def forward(self, state):
        """Return action probabilities."""
        hidden = self.layer1.forward(state)
        logits = self.layer2.forward(hidden)
        return logits
    
    def parameters(self):
        """Get trainable parameters."""
        return self.layer1.weights + self.layer2.weights
```

### Example: Training with Optimizer
```python
import euclid

optimizer = euclid.SGD(learning_rate=0.001)

# Forward pass
logits = network.forward(state)

# Compute loss and gradients
loss = compute_policy_loss(logits, actions, advantages)
gradients = loss.backward()

# Update parameters
optimizer.step(network.parameters(), gradients)
```

## Integration Points in Archid

### 1. Actor-Critic Base Class
File: `archid/reinforcement_learning/algorithms/actor_critic/actor_critic.py`

Expected methods:
- `actor.forward(state)` → returns action/policy
- `critic.forward(state)` → returns value estimate
- `actor_optimizer.step()` → updates actor weights
- `critic_optimizer.step()` → updates critic weights

### 2. PPO Implementation
File: `archid/reinforcement_learning/algorithms/actor_critic/v_critic/ppo.py`

Current implementation:
```python
def predict(self, state):
    distribution = self.actor.forward(state)
    return distribution.sample()

def update(self, experience):
    # Compute advantages and policy loss
    # Use actor/critic optimizers to update
    pass
```

### 3. Experience/Transition Data
File: `archid/reinforcement_learning/experience/experience.py`

Provides trajectory data for learning:
```python
class Transition:
    state: int or array
    action: int
    reward: float
    next_state: int or array
    done: bool
    log_prob: float  # For policy gradient methods
```

## Comparison: Tabular vs Neural Network Approaches

| Aspect | Tabular (Q-Learning) | Neural (PPO) |
|--------|----------------------|--------------|
| State Representation | Integer index | Neural network embedding |
| Storage | Dictionary Q-table | Network weights |
| Scalability | Up to ~1M states | High-dimensional spaces |
| Convergence | Fast on small problems | Slower, needs more data |
| Euclid Dependency | ❌ No | ✅ Yes |
| Examples | ✅ Cliff Walking | Need proper networks |

## Future Integration Goals

### Phase 1: Stabilize PPO (In Progress)
- ✅ Fix circular import bug
- ✅ Implement mock networks for testing
- ⏳ Replace with proper Euclid networks
- ⏳ Optimize hyperparameters

### Phase 2: Implement Deep Methods
- DQN with experience replay
- Double DQN for stability
- Dueling architectures
- Prioritized experience replay

### Phase 3: Add Advanced Algorithms
- A2C/A3C for parallel training
- DDPG for continuous control
- TD3 for robust learning
- SAC for entropy-regularized policies

### Phase 4: Production Quality
- Proper error handling
- Tensorboard logging
- Checkpoint/restore functionality
- Distributed training support

## Best Practices

### When to Use Which Algorithm

**Use Q-Learning/SARSA/TD when:**
- State space is discrete and small (< 100K states)
- Action space is discrete
- Fast convergence needed
- No neural networks required
- ✅ Archid provides excellent support

**Use PPO/Actor-Critic when:**
- State space is continuous or high-dimensional
- Action space is continuous or very large
- Complex feature extraction needed
- ⏳ Archid support being enhanced

### Integration Checklist
- [ ] Install euclid-ml >= 0.9.9
- [ ] Import euclid in actor-critic algorithms
- [ ] Define network architecture using euclid layers
- [ ] Implement forward pass (state → action/value)
- [ ] Implement gradient computation
- [ ] Connect to optimizer for updates
- [ ] Test on simple environments first
- [ ] Benchmark against tabular baselines

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'euclid'"
**Solution:** Install euclid-ml
```bash
pip install euclid-ml>=0.9.9
```

### Issue: "AttributeError: 'float' object has no attribute 'backward'"
**Solution:** Return proper Euclid tensor from forward pass
```python
# Wrong:
return np.array([0.5])

# Right:
return euclid.tensor([0.5])
```

### Issue: Gradients not updating weights
**Solution:** Ensure optimizer.step() uses returned gradients
```python
gradients = loss.backward()
optimizer.step(gradients)  # Pass gradients to optimizer
```

## Resources
- Euclid ML GitHub: https://github.com/euclid-ml/euclid
- Archid Docs: See README.md
- RL Theory: Reinforcement Learning: An Introduction (Sutton & Barto)
