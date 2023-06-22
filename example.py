"""
main script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
author: Cyril Hsu @ UvA-MNS
date: 23/02/2023
"""

import numpy as np
from tqdm import tqdm
from agents import baselineAgent
from rofarsEnv import ROFARS_v1

np.random.seed(0)

env = ROFARS_v1()
best_theta = None
best_total_reward = -np.inf

# training
for theta in range(5):
    env.reset(mode='train')
    agent = baselineAgent(theta=theta)
    # give random scores as the initial action
    init_action = np.random.rand(env.n_camera)
    reward, state, stop = env.step(init_action)

    for t in tqdm(range(env.length), initial=2):

        action = agent.get_action(state)
        reward, state, stop = env.step(action)

        # do sth to update your algorithm here

        if stop:
            break

    total_reward = env.get_total_reward()
    if total_reward > best_total_reward:
        best_theta = theta
        best_total_reward = total_reward

    print(f'=== TRAINING theta: {theta} ===')
    print('[total reward]:', total_reward)

print(f'Best found theta: {best_theta}')

# testing
env.reset(mode='test')
agent = baselineAgent(theta=best_theta)
# give random scores as the initial action
init_action = np.random.rand(env.n_camera)
reward, state, stop = env.step(init_action)

for t in tqdm(range(env.length), initial=2):

    action = agent.get_action(state)
    reward, state, stop = env.step(action)


    print("action", action)
    print("state", state)
    print("reward", reward)

    if stop:
        break

print(f'====== TESTING ======')
print('[total reward]:', env.get_total_reward())