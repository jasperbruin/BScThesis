import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import UCBAgent, SlidingWindowUCBAgent
#
# np.random.seed(0)
# env = ROFARS_v1()
# agent = UCBAgent(c=3)
# n_episode = 30
#
# # Initialize the UCB Agent
# agent.initialize(env.n_camera)
#
# # training
# for episode in range(n_episode):
#     env.reset(mode='train')
#
#     for t in tqdm(range(env.length), initial=2):
#         action = agent.get_action()
#         reward, state, stop = env.step(action)
#
#         # Update the UCB Agent
#         agent.update(action, state)
#
#         if stop:
#             break
#
#     print(f'=== TRAINING episode {episode} ===')
#     print('[total reward]:', env.get_total_reward())
#
# # testing
# env.reset(mode='test')
#
# for t in tqdm(range(env.length), initial=2):
#     action = agent.get_action()
#     reward, state, stop = env.step(action)
#
#     if stop:
#         break
#
# print(f'====== TESTING ======')
# print('[total reward]:', env.get_total_reward())

np.random.seed(0)
env = ROFARS_v1()

n_episode = 30
max_window_size = 100
best_window_size = 1
best_reward = -np.inf

# TODO: Sliding window has to go over training loop not over testing loop
for window_size in range(1, max_window_size + 1):
    agent = SlidingWindowUCBAgent(c=3, window_size=window_size)
    agent.initialize(env.n_camera)

    # training
    for episode in range(n_episode):
        env.reset(mode='train')

        for t in tqdm(range(env.length), initial=2):
            action = agent.get_action()
            reward, state, stop = env.step(action)

            # Update the UCB Agent
            agent.update(action, state)

            if stop:
                break

        print(f'=== TRAINING episode {episode}, window size {window_size} ===')
        print('[total reward]:', env.get_total_reward())

    # testing
    env.reset(mode='test')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        if stop:
            break

    total_reward = env.get_total_reward()
    print(f'====== TESTING window size {window_size} ======')
    print('[total reward]:', total_reward)

    # Save the best window size and total reward
    if total_reward > best_reward:
        best_reward = total_reward
        best_window_size = window_size

print(f'Best window size: {best_window_size}')
print(f'Best [total reward]: {best_reward}')