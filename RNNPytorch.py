import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from sklearn.metrics import mean_squared_error
from agents import baselineAgent, LSTM_Agent
import matplotlib.pyplot as plt

device = torch.device("mps")

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
        X.append(states[i: (i + time_steps)])
        Y.append(states[i + time_steps])
    return np.array(X), np.array(Y)


def impute_missing_values(states):
    imputed_states = []
    for state in states:
        mean_values = np.mean([v for v in state if v >= 0])
        imputed_state = np.array([v if v >= 0 else mean_values for v in state])
        imputed_states.append(imputed_state)
    return np.array(imputed_states)


class LSTM_Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_Agent, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def create_training_traces(env, mode='train'):
    # Training
    env.reset(mode)
    baseline_agent = baselineAgent()
    states = []

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


# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title(
        'Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
    plt.show()


def print_error(trainY, testY, train_predict, test_predict):
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))


np.random.seed(0)

env = ROFARS_v1()
best_total_reward = -np.inf

input_size = env.n_camera
hidden_size = 32
time_steps = 50
output_size = env.n_camera
lstm_agent = LSTM_Agent(input_size, hidden_size, output_size)
criterion = nn.MSELoss()

# Training
train_data = create_training_traces(env, mode='train')
train_data = impute_missing_values(train_data)
train_states, test_states = get_train_test(train_data, split_percent=0.8)
trainX, trainY = get_XY(train_states, time_steps)
testX, testY = get_XY(test_states, time_steps)

# Convert to torch tensors
trainX = torch.from_numpy(trainX).type(torch.Tensor)
trainY = torch.from_numpy(trainY).type(torch.Tensor)
testX = torch.from_numpy(testX).type(torch.Tensor)

# Train the model
optimizer = torch.optim.Adam(lstm_agent.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    lstm_agent.train()
    outputs = lstm_agent(trainX)
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# Test the model
lstm_agent.eval()
train_predict = lstm_agent(trainX)
test_predict = lstm_agent(testX)

# Invert predictions
train_predict = train_predict.detach().numpy()
trainY = trainY.detach().numpy()
test_predict = test_predict.detach().numpy()
testY = testY.detach().numpy()

# Plot the result
plot_result(trainY, testY, train_predict, test_predict)