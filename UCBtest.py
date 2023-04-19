"""
Suboptimal hyperparameters: The exploration parameter c and the window size might not be tuned correctly for this problem. You may need to perform hyperparameter tuning to find optimal values for both these parameters.
Randomness: There is randomness in the action selection process when self.counts.min() == 0. This means that when there is a tie, a random action is selected, which could lead to suboptimal choices.
Implementation issues: There might be a bug in the get_action method. In the else block, the action should be selected based on the highest UCB value, but instead, it is assigned the ucb_values array itself. This should be changed to:
Non-stationary environment: The ROFARS environment is likely non-stationary, meaning the optimal actions may change over time. The SlidingWindowUCBAgent may not be able to adapt quickly enough to these changes, resulting in a lower total reward.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import SlidingWindowUCBAgent

np.random.seed(0)
env = ROFARS_v1()
max_window_size = 100
best_window_size = 1
best_reward = -np.inf

window_sizes = []
total_rewards = []

# Find the best sliding window in the training session
for window_size in range(1, max_window_size + 1):
    agent = SlidingWindowUCBAgent(c=3, window_size=window_size * 60)
    agent.initialize(env.n_camera)

    # Training loop
    env.reset(mode='train')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        if stop:
            break

    total_reward = env.get_total_reward()
    print(f'=== TRAINING window size {window_size} ===')
    print('[total reward]:', total_reward)

    # Save the best window size and total reward
    if total_reward > best_reward:
        best_reward = total_reward
        best_window_size = window_size

    # Record the window size and its total reward
    window_sizes.append(window_size)
    total_rewards.append(total_reward)

# Use the best sliding window for testing
agent = SlidingWindowUCBAgent(c=3, window_size=best_window_size)
agent.initialize(env.n_camera)

env.reset(mode='test')

for t in tqdm(range(env.length), initial=2):
    action = agent.get_action()
    reward, state, stop = env.step(action)

    # Update the UCB Agent
    agent.update(action, state)

    if stop:
        break

test_total_reward = env.get_total_reward()
print(f'====== TESTING window size {best_window_size} ======')
print('[total reward]:', test_total_reward)
print(f'Best window size: {best_window_size}')
print(f'Best [total reward]: {best_reward}')

# Plot the window size and its total reward
plt.plot(window_sizes, total_rewards, label=f"Best window size: {best_window_size}, Total reward: {best_reward:.3f}")
plt.xlabel('Window Size', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('Sliding Window UCB: Window Size vs Total Reward', fontsize=14)
plt.legend(fontsize=10)
plt.grid()
plt.tight_layout()
plt.savefig('UCB.png')
plt.show()