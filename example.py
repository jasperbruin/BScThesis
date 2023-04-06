# main script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023

import numpy as np
from tqdm import tqdm
from agents import baselineAgent, LSTMAgent
from rofarsEnv import ROFARS_v1
import LSTM as LSTM
import torch
np.random.seed(0)

env = ROFARS_v1() # Use 2 hours of data for training

model = LSTM.LSTM(input_size=env.n_camera, hidden_size=env.n_camera, num_layers=1, num_classes=env.n_camera)

agent = LSTMAgent(model=model, input_size=env.n_camera, record_length=10)

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
        input_tensor = torch.tensor([agent.records])
        # Pass the input tensor to the model and get the output tensor
        output_tensor = model(input_tensor)
        # Clear the previous records
        agent.clear_records()
        # Add the new record to the agent's record
        agent.add_record(output_tensor)

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
