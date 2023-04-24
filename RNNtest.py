import math

import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from tensorflow.keras.optimizers import Adam
from agents import baselineAgent, LSTM_Agent
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
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
output_size = env.n_camera
lstm_agent = LSTM_Agent(input_size, hidden_size, output_size)

optimizer = Adam(learning_rate=0.001)
lstm_agent.compile(optimizer, loss='mae')

time_steps = 10
train_data = create_training_traces(env, mode='train')

test_data = create_training_traces(env, mode='test')

train_data = impute_missing_values(train_data)
test_data = impute_missing_values(test_data)


trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)

lstm_agent.fit(trainX, trainY, epochs=5, batch_size=32, verbose=2)

# make predictions
train_predict = lstm_agent.predict(trainX)
test_predict = lstm_agent.predict(testX)

# Print error
print_error(trainY, testY, train_predict, test_predict)

# Plot result
plot_result(trainY, testY, train_predict, test_predict)

