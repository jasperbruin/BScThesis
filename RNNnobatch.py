import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import baselineAgent, LSTM_Agent, DiscountedUCBAgent, SlidingWindowUCBAgent
import torch
from torch import nn
from torch.optim import Adam
import csv

# Function to set the device to CUDA if available
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("mps")


baseline_agent = None
agent = None

# Hyperparameters
# 0.005
l_rate = 0.001
hidden_size = 10
time_steps = [3*60]
epochs = 5000
patience = 5

best_val_loss = float('inf')
epochs_without_improvement = 0
result = []

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

def imv(state):
    mean_value = np.mean([v for v in state if v >= 0])
    imputed_state = np.array([v if v >= 0 else mean_value for v in state])
    return imputed_state


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
    elif inp == 3:
        states = []
        agent = SlidingWindowUCBAgent(window_size=9*60)
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


if __name__ == '__main__':
    inp1 = int(input("1. MSE\n2. MAE \n3. Huber\n"))
    if inp1 == 1:
        criterion = nn.MSELoss()
    if inp1 == 2:
        criterion = nn.L1Loss()
    if inp1 == 3:
        criterion = nn.HuberLoss()

    np.random.seed(0)

    env = ROFARS_v1()

    input_size = env.n_camera
    output_size = env.n_camera
    inp2 = int(input("1. Baseline Agent 2. D-UCB Agent: 3. SW-UCB Agent\n"))


    train_data = create_training_traces(env, 'train', inp2)
    test_data = create_training_traces(env, 'test', inp2)

    train_data = impute_missing_values(train_data)
    test_data = impute_missing_values(test_data)

    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size).to(device)
    optimizer = Adam(lstm_agent.parameters(), lr=l_rate)

    for ts in time_steps:
        trainX, trainY = get_XY(train_data, ts)
        testX, testY = get_XY(test_data, ts)

        trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
        trainY = torch.tensor(trainY, dtype=torch.float32).to(device)
        testX = torch.tensor(testX, dtype=torch.float32).to(device)
        testY = torch.tensor(testY, dtype=torch.float32).to(device)

        # Training loop
        print('Training LSTM Agent')
        for epoch in range(epochs):
            hidden_state, cell_state = lstm_agent.init_hidden_cell_states(
                batch_size=trainX.size(0))
            optimizer.zero_grad()
            outputs, (hidden_state, cell_state) = lstm_agent(trainX, (
            hidden_state, cell_state))
            loss = criterion(outputs, trainY)
            loss.backward()
            optimizer.step()

            # Validation
            val_outputs, (_, _) = lstm_agent(testX,
                                             lstm_agent.init_hidden_cell_states(
                                                 batch_size=testX.size(0)))
            val_loss = criterion(val_outputs, testY)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f'Epoch: {epoch + 1}, Loss: {round(loss.item(), 3)}, Validation Loss: {round(val_loss.item(), 3)}')

            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

        # Testing loop
        print('Testing LSTM Agent')
        env.reset(mode='test')
        # give random scores as the initial action
        init_action = np.random.rand(env.n_camera)
        reward, state, stop = env.step(init_action)

        # Initialize the hidden and cell states for the LSTM agent
        hidden_state, cell_state = lstm_agent.init_hidden_cell_states(
            batch_size=1)
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)

        for t in tqdm(range(env.length), initial=2):
            # Prepare the input state for the LSTM agent
            input_state = torch.tensor(state, dtype=torch.float32).unsqueeze(
                0).unsqueeze(0).to(
                device)  # Add the batch and sequence dimensions

            # Get the action from the LSTM agent, passing the hidden and cell states
            action, (hidden_state, cell_state) = lstm_agent(input_state, (
            hidden_state, cell_state))
            action = action.squeeze().detach().cpu().numpy()


            # Perform the action in the environment
            reward, state, stop = env.step(action)
            state = impute_missing_values([state])[0]

            if stop:
                break

        print(f'====== RESULT ======')
        if inp2 == 1:
            print("Used Historical traces: Baseline Agent")
        if inp2 == 2:
            print("Used Historical traces: D-UCB Agent")
        if inp2 == 3:
            print("Used Historical traces: SW-UCB Agent")
        print('[total reward]:', env.get_total_reward())
        print('[Hyperparameters]')
        print("epochs: {} lr: {} \nhidden_size: {} time_steps: {} loss function: {}".format(epochs, l_rate, hidden_size, ts, inp1))


        total_reward = env.get_total_reward()

        # used historical trace, total reward, epochs, l_rate, hidden_size, amount of timesteps, 1: MSE, 2: MAE, 3: Huber
        result.append([inp2, total_reward, epochs, l_rate, hidden_size, ts, inp1])

        with open('results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in result:
                writer.writerow(row)

"""
====== RESULT ======
Used Historical traces: Baseline Agent
[total reward]: 0.534
[Hyperparameters]
epochs: 5000 lr: 0.001 
hidden_size: 16 time_steps: 60 loss function: 1

====== RESULT ======
Used Historical traces: D-UCB Agent
[total reward]: 0.509
[Hyperparameters]
epochs: 5000 lr: 0.001 
hidden_size: 16 time_steps: 60 loss function: 1

====== RESULT ======
Used Historical traces: SW-UCB Agent
[total reward]: 0.502
[Hyperparameters]
epochs: 5000 lr: 0.001 
hidden_size: 16 time_steps: 60 loss function: 1


====== RESULT ======
Used Historical traces: Baseline Agent
[total reward]: 0.387
[Hyperparameters]
epochs: 5000 lr: 0.001 
hidden_size: 1 time_steps: 540 loss function: 1


====== RESULT ======
Used Historical traces: Baseline Agent
[total reward]: 0.522
[Hyperparameters]
epochs: 5000 lr: 0.001 
hidden_size: 32 time_steps: 60 loss function: 1


====== RESULT ======
Used Historical traces: D-UCB Agent
[total reward]: 0.491
[Hyperparameters]
epochs: 10000 lr: 0.001 
hidden_size: 32 time_steps: 60 loss function: 1

====== RESULT ======
Used Historical traces: SW-UCB Agent
[total reward]: 0.494
[Hyperparameters]
epochs: 10000 lr: 0.001 
hidden_size: 32 time_steps: 60 loss function: 1

"""
