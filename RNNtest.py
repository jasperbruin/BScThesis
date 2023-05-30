import statistics
import time
import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import baselineAgent, LSTM_Agent, DiscountedUCBAgent, SlidingWindowUCBAgent, UCBAgent
import torch
from torch import nn
from torch.optim import Adam
import csv
from random import shuffle
from sklearn.utils import resample
import matplotlib.pyplot as plt


if not torch.backends.mps.is_available():
    reason = "MPS not available because the current PyTorch install was not built with MPS enabled." \
        if not torch.backends.mps.is_built() \
        else "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
    print(reason)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("mps")



baseline_agent = None
agent = None

l_rate = 0.001
hidden_size = 16
time_steps = 60
epochs = 5000
patience = 10

criterion = nn.MSELoss()
np.random.seed(0)
env = ROFARS_v1()

best_val_loss = float('inf')
epochs_without_improvement = 0
result = []
training_losses = []
validation_losses = []

def create_training_traces(env, mode, inp):
    # Training
    env.reset(mode)
    states = []

    if inp == 1:
        agent = baselineAgent(agent_type='strong')
        init_action = np.random.rand(env.n_camera)
        reward, state, stop = env.step(init_action)
    elif inp == 2:
        agent = DiscountedUCBAgent(gamma=0.999)
        agent.initialize(env.n_camera)
    elif inp == 3:
        agent = SlidingWindowUCBAgent(window_size=9*60)
        agent.initialize(env.n_camera)
    elif inp == 4:
        agent = UCBAgent()
        agent.initialize(env.n_camera)
    elif inp == 5:
        agent = baselineAgent(agent_type='simple')
        init_action = np.random.rand(env.n_camera)
        reward, state, stop = env.step(init_action)

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action(state) if inp in [1, 5] else agent.get_action()
        reward, state, stop = env.step(action)

        if inp in [2, 3, 4]:  # UCB agents
            agent.update(action, state)
        states.append(state)

        if stop:
            break

    return states

def get_train_test(states, split_percent=0.7):
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

def impute_missing_values(states, style):
    imputed_states = []
    if style == 1:
        for state in states:
            median_values = np.median([v for v in state if v >= 0])
            imputed_state = np.array([v if v >= 0 else median_values for v in state])
            imputed_states.append(imputed_state)
        return np.array(imputed_states)
    else:
        for state in states:
            mask = state >= 0
            x = np.where(mask)[0]
            y = state[mask]
            imputed_state = np.interp(np.arange(len(state)), x, y)
            imputed_states.append(imputed_state)
        return np.array(imputed_states)

def imv(state, style='median'):
    if style == 1:
        median_value = np.median([v for v in state if v >= 0])
        imputed_state = np.array([v if v >= 0 else median_value for v in state])
        return imputed_state
    else:
        mask = state >= 0
        x = np.where(mask)[0]
        y = state[mask]
        imputed_state = np.interp(np.arange(len(state)), x, y)
        return imputed_state


def resample_data(X, Y):
    X_resampled, Y_resampled = resample(X, Y, replace=True, n_samples=len(X) // 2, random_state=123)
    c = list(zip(X_resampled, Y_resampled))
    shuffle(c)
    X_resampled, Y_resampled = zip(*c)
    return X_resampled, Y_resampled


if __name__ == '__main__':
    used_agent = int(input("1. Baseline Strong 2. D-UCB Agent: 3. SW-UCB Agent 4. UCB-1 Agent 5.Baseline Simple\n"))
    imp = int(input("Imputation style: 1. Median 2. Interpolation\n"))


    train_data = create_training_traces(env, 'train', used_agent)
    test_data = create_training_traces(env, 'test', used_agent)

    train_data = impute_missing_values(train_data, style=imp)
    test_data = impute_missing_values(test_data, style=imp)

    lstm_agent = LSTM_Agent(env.n_camera, hidden_size, env.n_camera).to(device)
    optimizer = Adam(lstm_agent.parameters(), lr=l_rate)

    trainX, trainY = get_XY(train_data, time_steps)
    testX, testY = get_XY(test_data, time_steps)
    trainX, trainY = resample_data(trainX, trainY)

    # convert to np array
    trainX = np.array(trainX)
    trainY = np.array(trainY)



    trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
    trainY = torch.tensor(trainY, dtype=torch.float32).to(device)
    testX = torch.tensor(testX, dtype=torch.float32).to(device)
    testY = torch.tensor(testY, dtype=torch.float32).to(device)

    # Training loop
    print('Training LSTM Agent')
    for epoch in range(epochs):
        hidden_state, cell_state = lstm_agent.init_hidden_cell_states(
            batch_size=trainX.size(0))
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
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
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        val_loss = criterion(val_outputs, testY)
        validation_losses.append(round(val_loss.item(), 3))
        training_losses.append(round(loss.item(), 3))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_epoch = epoch
        else:
            epochs_without_improvement += 1


        print(
            f'Epoch: {epoch + 1}, Training Loss: {round(loss.item(), 3)}, Validation Loss: {round(val_loss.item(), 3)}')

        if epochs_without_improvement >= patience:
            print("Early stopping")
            break

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

    inference_times = []

    reward_on_time = []
    for t in tqdm(range(env.length), initial=2):
        # Prepare the input state for the LSTM agent
        # print(state)
        input_state = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).to(
            device)  # Add the batch and sequence dimensions

        # Measure inference time
        start_time = time.time()

        # Get the action from the LSTM agent, passing the hidden and cell states
        action, (hidden_state, cell_state) = lstm_agent(input_state, (
        hidden_state, cell_state))

        end_time = time.time()

        # Calculate and append inference time
        inference_time = (end_time - start_time) * 1000  # convert to ms
        inference_times.append(inference_time)

        action = action.squeeze().detach().cpu().numpy()

        # Perform the action in the environment
        reward, state, stop = env.step(action)
        reward_on_time.append(reward)
        state = impute_missing_values([state], imp)[0]


        if stop:
            break

    average_inference_time = statistics.mean(inference_times)

    print(f'====== RESULT ======')
    if used_agent == 1:
        print("Used Historical traces: Baseline Strong")
    if used_agent == 2:
        print("Used Historical traces: D-UCB Agent")
    if used_agent == 3:
        print("Used Historical traces: SW-UCB Agent")
    if used_agent == 4:
        print("Used Historical traces: UCB-1 Agent")
    if used_agent == 5:
        print("Used Historical traces: Baseline Weak")

    print('[total reward]:', env.get_total_reward())
    print('[Hyperparameters]')
    print(
        "epochs: {} l_rate: {} \nhidden_size: {} time_steps: {}".format(
            epochs, l_rate, hidden_size, time_steps))

    total_reward = env.get_total_reward()
    result.append(
        [used_agent, total_reward, best_epoch, epochs, l_rate, hidden_size, time_steps, average_inference_time])

    with open('results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in result:
            writer.writerow(row)

    # Plot the validation and training losses
    plt.plot(range(len(validation_losses)), validation_losses)
    plt.plot(range(len(training_losses)), training_losses)
    plt.legend(['Validation Loss', 'Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation and Training Losses')

    plt.savefig('losses.png')
    plt.show()