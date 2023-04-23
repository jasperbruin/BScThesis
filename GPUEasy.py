import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

device = torch.device("mps")

class DiscountedUCBAgent:
    def __init__(self, gamma=0.9):
        self.counts = None
        self.discounted_counts = None
        self.values = None
        self.discounted_rewards = None
        self.c = 3
        self.gamma = gamma
        self.total_time_steps = 0

    def initialize(self, n_actions):
        self.counts = np.zeros(n_actions)
        self.discounted_counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)
        self.discounted_rewards = np.zeros(n_actions)

    def get_action(self):
        if self.counts.min() == 0:
            idx = np.random.choice(np.where(self.counts == 0)[0])
            action = np.zeros(len(self.values))
            action[idx] = 1
        else:
            discounted_means = self.discounted_rewards / self.discounted_counts
            ct_numerator = 2 * np.log(self.total_time_steps)
            ct_denominator = self.discounted_counts
            ct = self.c * np.sqrt(np.maximum(ct_numerator / ct_denominator, 0))
            action = discounted_means + ct
        return action

    def update(self, actions, state):
        self.total_time_steps = self.gamma*self.total_time_steps + 1
        self.discounted_counts *= self.gamma
        self.discounted_rewards *= self.gamma

        for i, reward in enumerate(state):
            if reward >= 0:
                self.counts[i] += 1
                self.values[i] += reward

                self.discounted_counts[i] += 1
                self.discounted_rewards[i] += reward
            else:
                self.counts[i] += 0


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




def evaluateUCB():
    lr = float(input('Learning rate: '))
    epochs = int(input('Epochs: '))
    hidden_size = int(input('Hidden size: '))

    env = ROFARS_v1()
    best_total_reward = -np.inf
    input_size = env.n_camera
    output_size = env.n_camera
    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(lstm_agent.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training
    baseline_agent = DiscountedUCBAgent(gamma=0.999)
    baseline_agent.initialize(env.n_camera)
    env.reset(mode='train')

    states, actions = [], []

    # Generate training traces from the Baseline agent
    init_action = np.random.rand(env.n_camera)
    reward, state, stop = env.step(init_action)

    for t in tqdm(range(env.length), initial=2):
        action = baseline_agent.get_action()
        reward, state, stop = env.step(action)
        baseline_agent.update(action, state)


        states.append(state)
        actions.append(action)

        if stop:
            break

    # Train the LSTM agent with the training traces
    states = np.array(states)
    actions = np.array(actions)

    states = states.reshape((states.shape[0], 1, states.shape[1]))
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = lstm_agent(states_tensor)
        loss = criterion(predictions, actions_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.3f}')

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
    print('[total reward]:', f'{total_reward:.3f}')

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
    print('[total reward]:', f'{np.mean(rewards):.3f}')


def evaluateBaseline():
    lr = float(input('Learning rate: '))
    epochs = int(input('Epochs: '))
    hidden_size = int(input('Hidden size: '))

    env = ROFARS_v1()
    best_total_reward = -np.inf
    input_size = env.n_camera
    output_size = env.n_camera
    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(lstm_agent.parameters(), lr=lr)
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

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = lstm_agent(states_tensor)
        loss = criterion(predictions, actions_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.3f}')

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
    print('[total reward]:', f'{total_reward:.3f}')

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
    print('[total reward]:', f'{np.mean(rewards):.3f}')

if __name__ == '__main__':
    inp = int(input('1. Baseline\n2. UCB\n'))
    if inp == 1:
        evaluateBaseline()
    elif inp == 2:
        evaluateUCB()


"""
Run 1: baseline
Learning rate: >? 0.001
Epochs: >? 2400
Hidden size: >? 64
Loss: 0.968
[total reward]: 0.440                      
====== TESTING ======
[total reward]: 0.430
"""