# main script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023

import numpy as np
from tqdm import tqdm
from agents import baselineAgent
from rofarsEnv import ROFARS_v1
from sklearn.model_selection import train_test_split


np.random.seed(0)

env = ROFARS_v1()
agent = baselineAgent(36*10)
n_episode = 30

# training
for episode in range(n_episode):

    env.reset(mode='train')
    agent.clear_records()
    # give random scores as the initial action
    init_action = np.random.rand(env.n_camera)
    reward, state, stop = env.step(init_action)

    for t in tqdm(range(env.length), initial=2):

        action = agent.get_action(state)
        reward, state, stop = env.step(action)

        # do sth to update your algorithm here

        if stop:
            break

    print(f'=== TRAINING episode {episode} ===')
    print('[total reward]:', env.get_total_reward())

# testing
env.reset(mode='test')
agent.clear_records()
# give random scores as the initial action
init_action = np.random.rand(env.n_camera)
reward, state, stop = env.step(init_action)

for t in tqdm(range(env.length), initial=2):

    action = agent.get_action(state)
    reward, state, stop = env.step(action)

    if stop:
        break

print(f'====== TESTING ======')
print('[total reward]:', env.get_total_reward())