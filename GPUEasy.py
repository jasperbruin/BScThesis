from rofarsEnv import ROFARS_v1
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

device = torch.device("mps")

class baselineAgent:
    def __init__(self, theta=0, agent_type='strong'):
        assert agent_type in ['simple', 'strong']
        self.agent_type = agent_type
        self.prev_state = None
        self.theta = theta

    def get_action(self, state):
        # store previous state
        self.prev_state = state

        if self.agent_type=='simple':
            # random action
            action = np.random.rand(len(state))

        elif self.agent_type=='strong':
            # use previous states as scores (-1 is replaced by the learned param theta)
            action = np.array([v if v>=0 else self.theta for v in self.prev_state])

        return action

class LSTM_Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_Agent, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

        self.records = [[] for _ in range(input_size)]

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]
        x = self.dense(x)
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
                imputed_state[i] = self.interpolate_missing_value(i)
            else:
                self.records[i].append(s)
        return imputed_state

    def interpolate_missing_value(self, index):
        if len(self.records[index]) < 2:
            return np.mean(self.records[index]) if len(self.records[index]) > 0 else 0

        x = np.arange(len(self.records[index]))
        y = np.array(self.records[index])
        spline_order = 3
        interpolator = interp1d(x, y, kind='linear', fill_value='extrapolate')
        return float(interpolator(len(self.records[index])))




def train_lstm_agent(agent, dataloader, criterion, optimizer, epochs):
    agent.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for states, actions in dataloader:
            optimizer.zero_grad()
            predictions = agent(states)
            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.3f}')


def test_lstm_agent(agent, env):
    agent.eval()
    env.reset(mode='test')
    rewards = []

    state = env.reset()
    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action(state)
        reward, state, stop = env.step(action)
        rewards.append(reward)

        if stop:
            break

    return np.mean(rewards)


def evaluateBaseline():
    lr = float(input('Learning rate: '))
    epochs = int(input('Epochs: '))
    hidden_size = int(input('Hidden size: '))
    batch_size = int(input('Batch size: '))
    loss_function = int(input('Loss function (1: MSE, 2: MAE): '))

    env = ROFARS_v1()
    input_size = env.n_camera
    output_size = env.n_camera
    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(lstm_agent.parameters(), lr=lr)

    if loss_function == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

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

    train_dataset = TensorDataset(states_tensor, actions_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_lstm_agent(lstm_agent, train_dataloader, criterion, optimizer, epochs)

    # Test the LSTM agent
    test_reward = test_lstm_agent(lstm_agent, env)
    print(f'====== TESTING ======')
    print('[total reward]:', f'{test_reward:.3f}')

if __name__ == '__main__':
    evaluateBaseline()


"""
Run 1:
Learning rate: >? 0.001
Epochs: >? 2400
Hidden size: >? 64
Loss: 0.968
[total reward]: 0.440                      
====== TESTING ======
[total reward]: 0.430

Run 2:
Learning rate: >? 0.001
Epochs: >? 10
Hidden size: >? 32
Batch size: >? 32
Loss function (1: MSE, 2: MAE): >? 1

"""