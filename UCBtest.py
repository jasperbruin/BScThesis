import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import SlidingWindowUCBAgent

np.random.seed(0)
env = ROFARS_v1()

n_episode = 1
max_window_size = 100
best_window_size = 1
best_reward = -np.inf

# Find the best sliding window in the training session
for window_size in range(1, max_window_size + 1):
    agent = SlidingWindowUCBAgent(c=3, window_size=window_size)
    agent.initialize(env.n_camera)

    # Training loop
    for episode in range(n_episode):
        env.reset(mode='train')

        for t in tqdm(range(env.length), initial=2):
            action = agent.get_action()
            reward, state, stop = env.step(action)

            # Update the UCB Agent
            agent.update(action, state)

            print("action: ", action)
            print("state: ", state)
            print("reward: ", reward)

            if stop:
                break

        total_reward = env.get_total_reward()
        print(f'=== TRAINING episode {episode}, window size {window_size} ===')
        print('[total reward]:', total_reward)

        # Save the best window size and total reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_window_size = window_size

# Use the best sliding window for testing
agent = SlidingWindowUCBAgent(c=3, window_size=best_window_size)
agent.initialize(env.n_camera)

env.reset(mode='test')

for t in tqdm(range(env.length), initial=2):
    action = agent.get_action()
    reward, state, stop = env.step(action)

    if stop:
        break

test_total_reward = env.get_total_reward()
print(f'====== TESTING window size {best_window_size} ======')
print('[total reward]:', test_total_reward)
print(f'Best window size: {best_window_size}')
print(f'Best [total reward]: {best_reward}')
