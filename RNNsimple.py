import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import baselineAgent, LSTM_Agent, DiscountedUCBAgent, SlidingWindowUCBAgent
import torch
from torch import nn
from torch.optim import Adam

device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_available() else "cpu")

baseline_agent = None
agent = None

batch_size = 32
l_rate = 0.001
hidden_size = 16
time_steps = 60
epochs = 1000

# Add these new variables before the training loop
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0


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
    inp = int(input("1. MSE\n2. MAE \n3. Huber\n"))
    if inp == 1:
        criterion = nn.MSELoss()
    if inp == 2:
        criterion = nn.L1Loss()
    if inp == 3:
        criterion = nn.HuberLoss()

    np.random.seed(0)

    env = ROFARS_v1()

    input_size = env.n_camera
    output_size = env.n_camera
    inp = int(input("1. Baseline Agent 2. D-UCB Agent: 3. SW-UCB Agent\n"))


    train_data = create_training_traces(env, 'train', inp)
    test_data = create_training_traces(env, 'test', inp)

    train_data = impute_missing_values(train_data)
    test_data = impute_missing_values(test_data)

    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size).to(device)
    optimizer = Adam(lstm_agent.parameters(), lr=l_rate)

    trainX, trainY = get_XY(train_data, time_steps)
    testX, testY = get_XY(test_data, time_steps)

    trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
    trainY = torch.tensor(trainY, dtype=torch.float32).to(device)
    testX = torch.tensor(testX, dtype=torch.float32).to(device)
    testY = torch.tensor(testY, dtype=torch.float32).to(device)

    # Training loop
    print('Training LSTM Agent')
    for epoch in range(epochs):
        # Training
        for i in range(0, len(trainX), batch_size):
            x_batch = trainX[i: i + batch_size]
            y_batch = trainY[i: i + batch_size]
            hidden_state, cell_state = lstm_agent.init_hidden_cell_states(
                batch_size=x_batch.size(0))
            optimizer.zero_grad()
            outputs, (hidden_state, cell_state) = lstm_agent(x_batch, (
                hidden_state, cell_state))
            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()
            loss = criterion(outputs, y_batch)
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
    hidden_state, cell_state = lstm_agent.init_hidden_cell_states(batch_size=1)

    for t in tqdm(range(env.length), initial=2):
        # Prepare the input state for the LSTM agent
        input_state = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).to(device)  # Add the batch and sequence dimensions

        # Get the action from the LSTM agent, passing the hidden and cell states
        action, (hidden_state, cell_state) = lstm_agent(input_state, (
        hidden_state, cell_state))
        action = action.squeeze().detach().numpy()

        # Perform the action in the environment
        reward, state, stop = env.step(action)
        state = imv(state)


        if stop:
            break

    print(f'====== RESULT ======')
    print('[total reward]:', env.get_total_reward())




"""
SW-UCB, MSE, 100 epochs, 0.001 lr, 64 hidden size, 30 time steps, 0.1 reward threshold
====== RESULT ======
[total reward]: 0.505
"""