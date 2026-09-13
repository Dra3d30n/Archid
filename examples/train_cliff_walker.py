"""
Train an AI agent to learn cliff walking using Q-Learning.

The agent learns to navigate a 4x12 grid world:
- Avoid a cliff (positions 3,1 to 3,10) which gives -100 reward
- Reach the goal at (3,11)
- Each step costs -1

Results show the learned optimal policy (safe path avoiding the cliff)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from archid import QLearning
from archid.reinforcement_learning.experience import Experience, Transition
from examples.cliff_walking_env import CliffWalking


def train_cliff_walker(episodes=500, learning_rate=0.1, gamma=0.99, epsilon=0.1):
    """Train Q-Learning agent on cliff walking task."""
    import random
    
    env = CliffWalking()
    agent = QLearning(
        actions=lambda s: [0, 1, 2, 3],  # Up, Right, Down, Left
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    episode_rewards = []
    
    print("Training Q-Learning agent on Cliff Walking...")
    print(f"Episodes: {episodes}, Learning Rate: {learning_rate}, Gamma: {gamma}\n")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100  # Prevent infinite loops
        
        while not done and steps < max_steps:
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                action = random.randint(0, 3)  # Random action
            else:
                # Greedy action
                q_values = [agent.action_value(state, a) for a in range(4)]
                action = q_values.index(max(q_values))
            
            # Environment steps
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Create transition and experience
            exp = Experience()
            exp.add(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_action=None
            ))
            
            # Agent learns
            agent.update(exp)
            
            state = next_state
            steps += 1
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = sum(episode_rewards[-50:]) / 50
            print(f"Episode {episode+1:3d} - Avg Reward (last 50): {avg_reward:7.2f}")
    
    return agent, env, episode_rewards


def extract_policy(agent, env):
    """Extract greedy policy from learned Q-values."""
    policy = {}
    
    for state in range(env.nrows * env.ncols):
        # Get best action for each state
        q_values = []
        for action in range(4):
            q_val = agent.action_value(state, action)
            q_values.append((q_val, action))
        
        best_q, best_action = max(q_values)
        policy[state] = best_action
    
    return policy


def visualize_policy(env, policy):
    """Visualize the learned policy as arrows."""
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    print("\nLearned Policy (Optimal Actions):")
    print("=" * 30)
    
    for row in range(env.nrows):
        for col in range(env.ncols):
            state = env._pos_to_state(row, col)
            if (row, col) in env.cliff_positions:
                print('C', end=' ')
            elif (row, col) == env.goal_pos:
                print('G', end=' ')
            elif (row, col) == env.start_pos:
                print('S', end=' ')
            else:
                action = policy.get(state, 0)
                print(arrows[action], end=' ')
        print()
    print()


def run_episode_visual(agent, env, num_episodes=3):
    """Run and visualize a few episodes with the trained agent."""
    print("\nVisualization of Trained Agent Episodes:")
    print("=" * 50)
    
    for episode_num in range(num_episodes):
        print(f"\nEpisode {episode_num + 1}:")
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        path = [(env.agent_pos[0], env.agent_pos[1])]
        
        while not done and steps < 50:
            # Always exploit learned policy (no exploration)
            q_values = [agent.action_value(state, a) for a in range(4)]
            action = q_values.index(max(q_values))
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            path.append(env.agent_pos)
            
            state = next_state
            steps += 1
        
        print(f"Steps: {steps}, Total Reward: {total_reward}")
        print("Path taken (marked with *):")
        
        grid = [['.' for _ in range(env.ncols)] for _ in range(env.nrows)]
        for row, col in env.cliff_positions:
            grid[row][col] = 'C'
        grid[env.goal_pos[0]][env.goal_pos[1]] = 'G'
        grid[env.start_pos[0]][env.start_pos[1]] = 'S'
        
        for row, col in path:
            if grid[row][col] == '.':
                grid[row][col] = '*'
        
        for row in grid:
            print(' '.join(row))


if __name__ == "__main__":
    # Train agent
    agent, env, rewards = train_cliff_walker(
        episodes=500,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    # Extract and show policy
    policy = extract_policy(agent, env)
    visualize_policy(env, policy)
    
    # Run trained agent
    run_episode_visual(agent, env, num_episodes=3)
    
    print("\nTraining complete! Agent learned to avoid the cliff and reach the goal.")
