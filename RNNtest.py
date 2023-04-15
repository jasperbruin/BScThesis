# main script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023

from agents import LSTM_RNN_Agent, SlidingWindowUCBAgent
from rofarsEnv import ROFARS_v1
import numpy as np
from tqdm import tqdm

np.random.seed(0)
env = ROFARS_v1()

n_episode = 1
max_window_size = 100
best_window_size = 1
best_reward = -np.inf


# Collect necessary features for the LSTM RNN
train_states = []
train_rewards = []
train_actions = []

# Training loop
for episode in range(n_episode):
    agent = SlidingWindowUCBAgent(c=3, window_size=31)
    agent.initialize(env.n_camera)  # Make sure to initialize the agent
    env.reset(mode='train')
    episode_states = []
    episode_rewards = []
    episode_actions = []

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        # Save state, reward, and action
        episode_states.append(state)
        episode_rewards.append(reward)
        episode_actions.append(action)

        if stop:
            break

    train_states.append(episode_states)
    train_rewards.append(episode_rewards)
    train_actions.append(episode_actions)

    total_reward = env.get_total_reward()
    print(f'=== TRAINING episode {episode} ===')
    print('[total reward]:', total_reward)


# Save the training traces in a dictionary
training_trace = {
    'states': train_states,
    'rewards': train_rewards,
    'actions': train_actions
}

# Create the LSTM RNN Agent
lstm_agent = LSTM_RNN_Agent(record_length=env.n_camera, input_dim=env.n_camera, hidden_dim=32,
                            learning_rate=0.001, dropout_rate=0.0, l1_reg=0.0,
                            l2_reg=0.0, epsilon=0.1)

# Train the LSTM RNN Agent using the training_trace
n_samples = sum([len(episode_states) - 9 for episode_states in train_states])

X = np.concatenate([episode_states[i:i+10] for episode_states in train_states for i in range(len(episode_states) - 9)]).reshape(n_samples, 10, env.n_camera)
y = np.concatenate([episode_actions[i+9] for episode_actions in train_actions for i in range(len(episode_actions) - 9)]).reshape(n_samples, env.n_camera)

lstm_agent.train_on_batch(X, y)


# Testing loop
env.reset(mode='test')
test_rewards = []

for t in tqdm(range(env.length), initial=2):
    action = lstm_agent.get_action(state)
    reward, state, stop = env.step(action)

    test_rewards.append(reward)

    if stop:
        break

test_total_reward = np.sum(test_rewards)
print(f'====== TESTING LSTM RNN Agent ======')
print('[total reward]:', test_total_reward)