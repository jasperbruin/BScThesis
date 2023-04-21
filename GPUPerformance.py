import torch
import torch.nn as nn

class LSTM_Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_Agent, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.records = [[] for _ in range(input_size)]

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]
        x = self.dense(x)
        x = self.softmax(x)
        return x

    def get_action(self, state):
        imputed_state = self.impute_missing_values(state)
        state = np.expand_dims(imputed_state, axis=0)
        state = np.expand_dims(state, axis=1)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = self(state)
        return action.detach().cpu().numpy()[0]

    def impute_missing_values(self, state):
        imputed_state = np.copy(state)
        for i, s in enumerate(state):
            if s == -1:
                imputed_state[i] = np.mean(self.records[i]) if len(self.records[i]) > 0 else 0
            else:
                self.records[i].append(s)
        return imputed_state

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import baselineAgent, LSTM_Agent
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
env = ROFARS_v1()
best_total_reward = -np.inf

input_size = env.n_camera
hidden_size = 64
output_size = env.n_camera
lstm_agent = LSTM_Agent(input_size, hidden_size, output_size).to(device)


optimizer = optim.Adam(lstm_agent.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training
env.reset(mode='train')
baseline_agent = baselineAgent()
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
states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)

for epoch in range(1000):
    optimizer.zero_grad()
    predictions = lstm_agent(states_tensor)
    loss = criterion(predictions, actions_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

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
    best_total_reward = total_reward

print(f'=== TRAINING ===')
print('[total reward]:', total_reward)

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

