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
        self.relu = nn.ReLU()
        self.records = [[] for _ in range(input_size)]

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]
        x = self.dense(x)
        x = self.relu(x)
        return x

    def get_action(self, state):
        imputed_state = self.impute_missing_values(state)
        state = np.expand_dims(imputed_state, axis=0)
        state = np.expand_dims(state, axis=1)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = self(state)
        return action.detach().cpu().numpy()[0]

    def train_lstm_agent(self, dataloader, criterion, optimizer, epochs):
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for states, actions in dataloader:
                states = states.unsqueeze(1)
                optimizer.zero_grad()
                predictions = self(states)
                loss = criterion(predictions, actions)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(
                f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.3f}')

    def test_lstm_agent(self, dataloader, criterion):
        self.eval()
        running_loss = 0.0
        with torch.no_grad():
            for states, actions in dataloader:
                states = states.unsqueeze(1)
                predictions = self(states)
                loss = criterion(predictions, actions)
                running_loss += loss.item()
        return running_loss / len(dataloader)

def get_train_test(states, split_percent=0.8):
    n = len(states)
    split = int(n * split_percent)
    train_states = states[:split]
    test_states = states[split:]
    return train_states, test_states

def get_XY(states, time_steps=1):
    states = np.array(states)
    X, Y = [], []
    for i in range(len(states) - time_steps):
        X.append(states[i : (i + time_steps)])
        Y.append(states[i + time_steps])
    return np.array(X), np.array(Y)

def impute_missing_values(states):
    imputed_states = []
    for state in states:
        mean_values = np.mean([v for v in state if v >= 0])
        imputed_state = np.array([v if v >= 0 else mean_values for v in state])
        imputed_states.append(imputed_state)
    return np.array(imputed_states)



def create_training_traces(env, mode='train'):
    # Training
    env.reset(mode)
    baseline_agent = baselineAgent()
    states, actions = [], []

    # Generate training traces from the Baseline agent
    init_action = np.random.rand(env.n_camera)
    reward, state, stop = env.step(init_action)

    for t in tqdm(range(env.length), initial=2):
        action = baseline_agent.get_action(state)
        reward, state, stop = env.step(action)

        states.append(state)
        if stop:
            break

    return states

def impute_missing_values(state):
    # impute missing values with the mean if value is -1
    mean_values = np.mean([v for v in state if v >= 0])
    imputed_state = np.array([v if v >= 0 else mean_values for v in state])
    return imputed_state


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

    states = create_training_traces(env, mode='train')
    states = np.array(states)



    # Impute missing values
    states = impute_missing_values(states)

    print(states)

    # Split states into train and test
    train_states, test_states = get_train_test(states)

    # Prepare input X and target Y for training and testing
    X_train, Y_train = get_XY(train_states)
    X_test, Y_test = get_XY(test_states)



if __name__ == '__main__':
    evaluateBaseline()