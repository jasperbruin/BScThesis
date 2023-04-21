import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from tensorflow.keras.optimizers import Adam
from agents import baselineAgent, LSTM_Agent

np.random.seed(0)

env = ROFARS_v1()
best_theta = None
best_total_reward = -np.inf

input_size = env.n_camera
hidden_size = 16
output_size = env.n_camera
lstm_agent = LSTM_Agent(input_size, hidden_size, output_size)

optimizer = Adam(lr=0.001)
lstm_agent.compile(optimizer, loss='mse')

# Training
for theta in range(5):
    env.reset(mode='train')
    baseline_agent = baselineAgent(theta=theta)
    states, actions = [], []

    # Generate training traces from the Baseline agent
    init_action = np.random.rand(env.n_camera)
    reward, state, stop = env.step(init_action)

    for t in tqdm(range(env.length), initial=2):
        action = baseline_agent.get_action(state)
        states.append(state)
        actions.append(action)

        reward, state, stop = env.step(action)
        if stop:
            break

    # Train the LSTM agent with the training traces
    states = np.array(states)
    actions = np.array(actions)

    states = states.reshape((states.shape[0], 1, states.shape[1]))
    lstm_agent.fit(states, actions, epochs=10, verbose=1)

    # Test the LSTM agent
    env.reset(mode='test')
    rewards = []

    for t in tqdm(range(env.length), initial=2):
        action = lstm_agent.get_action(state)
        reward, state, stop = env.step(action)
        rewards.append(reward)

        if stop:
            break

    total_reward = np.mean(rewards)
    if total_reward > best_total_reward:
        best_theta = theta
        best_total_reward = total_reward

    print(f'=== TRAINING theta: {theta} ===')
    print('[total reward]:', total_reward)

print(f'Best found theta: {best_theta}')

# Testing
env.reset(mode='test')
rewards = []

for t in tqdm(range(env.length), initial=2):
    action = lstm_agent.get_action(state)
    reward, state, stop = env.step(action)
    rewards.append(reward)

    if stop:
        break

print(f'====== TESTING ======')
print('[total reward]:', np.mean(rewards))