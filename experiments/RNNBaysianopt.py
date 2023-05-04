import math
import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import baselineAgent, LSTM_Agent, DiscountedUCBAgent
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from bayes_opt import BayesianOptimization

batch_size = 32
criterion = None

inp = int(input("1. MSE\n2. MAE \n3. Huber\n"))
if inp == 1:
    criterion = nn.MSELoss()
if inp == 2:
    criterion = nn.L1Loss()
if inp == 3:
    criterion = nn.SmoothL1Loss()

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


def create_training_traces(env, mode, inp):

    # Training
    env.reset(mode)
    if inp == 1:
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
    elif inp == 2:
        states = []
        agent = DiscountedUCBAgent(gamma=0.999)
        agent.initialize(env.n_camera)

        for t in tqdm(range(env.length), initial=2):
            action = agent.get_action()
            reward, state, stop = env.step(action)

            # Update the UCB Agent
            agent.update(action, state)

            states.append(state)

            if stop:
                break

        return states


def train_and_evaluate(params):
    time_steps, epochs = map(int, params)

    hidden_size = 32

    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size)
    optimizer = Adam(lstm_agent.parameters(), lr=0.01)
    criterion = nn.L1Loss()

    trainX, trainY = get_XY(train_data, time_steps)
    testX, testY = get_XY(test_data, time_steps)

    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainY = torch.tensor(trainY, dtype=torch.float32)
    testX = torch.tensor(testX, dtype=torch.float32)
    testY = torch.tensor(testY, dtype=torch.float32)

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(trainX), batch_size):
            x_batch = trainX[i : i + batch_size]
            y_batch = trainY[i : i + batch_size]

            # Initialize the hidden and cell states for the LSTM agent with the correct batch size
            hidden_state, cell_state = lstm_agent.init_hidden_cell_states(
                batch_size=x_batch.size(0))

            optimizer.zero_grad()
            outputs, (hidden_state, cell_state) = lstm_agent(x_batch, (hidden_state, cell_state))
            # Detach the hidden and cell states to avoid backpropagating through the entire history
            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Initialize the hidden and cell states for the LSTM agent with the correct batch size
    hidden_state, cell_state = lstm_agent.init_hidden_cell_states(
        batch_size=trainX.size(0))
    # Make predictions on train data
    train_predict, _ = lstm_agent(trainX, (hidden_state, cell_state))
    train_predict = train_predict.detach().numpy()

    # Initialize the hidden and cell states for the LSTM agent with the correct batch size
    hidden_state, cell_state = lstm_agent.init_hidden_cell_states(
        batch_size=testX.size(0))
    # Make predictions on test data
    test_predict, _ = lstm_agent(testX, (hidden_state, cell_state))
    test_predict = test_predict.detach().numpy()

    # Print error
    train_rmse = math.sqrt(mean_squared_error(trainY.numpy(), train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY.numpy(), test_predict))

    return train_rmse, test_rmse



def optimize_lstm_agent(epochs, time_steps):
    _, test_rmse = train_and_evaluate((time_steps, epochs))
    return -test_rmse




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

def plot_rewards_comparison(ucb_rewards, lstm_rewards):
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(ucb_rewards, label='UCB Rewards')
    plt.plot(lstm_rewards, label='LSTM Rewards')
    plt.legend(['UCB Rewards', 'LSTM Rewards'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Rewards')
    plt.title('Comparison of UCB and LSTM Rewards')
    plt.show()



if __name__ == '__main__':
    np.random.seed(0)

    env = ROFARS_v1()

    input_size = env.n_camera
    output_size = env.n_camera
    inp = int(input("1. Baseline Agent 2. UCB Agent: "))

    train_data = create_training_traces(env, 'train', inp)
    test_data = create_training_traces(env, 'test', inp)

    train_data = impute_missing_values(train_data)
    test_data = impute_missing_values(test_data)

    pbounds = {
        'time_steps': (9*60, 20*60),
        'epochs': (1, 5)
    }

    optimizer = BayesianOptimization(
        f=optimize_lstm_agent,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize(init_points=5, n_iter=10)

    best_params = optimizer.max['params']
    hidden_size = 32
    time_steps = int(best_params['time_steps'])
    epochs = int(best_params['epochs'])

    print(
        f"Best parameters: Hidden size: {hidden_size}, Time steps: {time_steps}, Epochs: {epochs}")

    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size)
    optimizer = Adam(lstm_agent.parameters(), lr=0.01)

    trainX, trainY = get_XY(train_data, time_steps)
    testX, testY = get_XY(test_data, time_steps)

    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainY = torch.tensor(trainY, dtype=torch.float32)
    testX = torch.tensor(testX, dtype=torch.float32)
    testY = torch.tensor(testY, dtype=torch.float32)



    # Training loop
    print('Training LSTM Agent')
    for epoch in range(epochs):
        for i in range(0, len(trainX), batch_size):
            x_batch = trainX[i: i + batch_size]
            y_batch = trainY[i: i + batch_size]

            # Initialize the hidden and cell states for the LSTM agent with the correct batch size
            hidden_state, cell_state = lstm_agent.init_hidden_cell_states(
                batch_size=x_batch.size(0))

            optimizer.zero_grad()
            # Pass the hidden and cell states to the LSTM agent
            outputs, (hidden_state, cell_state) = lstm_agent(x_batch, (
            hidden_state, cell_state))
            # Detach the hidden and cell states to avoid backpropagating through the entire history
            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch + 1}, Loss: {round(loss.item(), 3)}')




    # Make predictions
    train_predict = lstm_agent(trainX).detach().numpy()
    test_predict = lstm_agent(testX).detach().numpy()

    # Print error
    print_error(trainY.numpy(), testY.numpy(), train_predict, test_predict)

    # Plot result
    plot_result(trainY.numpy(), testY.numpy(), train_predict, test_predict)



