# main script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023

import numpy as np
from tqdm import tqdm
from agents import SimpleRNNAgent
from rofarsEnv import ROFARS_v1


np.random.seed(0)
env = ROFARS_v1()
agent = SimpleRNNAgent(36 * 10)
n_episode = 30

# training
n_epochs = 5  # Reduced number of training epochs
model_update_frequency = 10  # Train the model after every 10 steps

for episode in range(n_episode):

    env.reset(mode='train')
    agent.clear_records()
    # give random scores as the initial action
    init_action = np.random.rand(env.n_camera)
    reward, state, stop = env.step(init_action)

    for t in tqdm(range(env.length), initial=2):

        action = agent.get_action(state)
        reward, state, stop = env.step(action)

        # Update your algorithm less frequently
        if t % model_update_frequency == 0:
            X = np.expand_dims(agent.records, axis=-1)
            y = np.expand_dims(state, axis=-1)
            agent.model.fit(X, y, epochs=n_epochs, verbose=0)

        if stop:
            break

    print(f'=== TRAINING episode {episode} ===')
    print('[total reward]:', env.get_total_reward())

# testing
agent.epsilon = 0  # Disable exploration
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