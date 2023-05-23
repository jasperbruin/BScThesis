import statistics
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import baselineAgent, LSTM_Agent, DiscountedUCBAgent, SlidingWindowUCBAgent, UCBAgent
import torch
from torch import nn
from torch.optim import Adam
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.utils import resample

# Function to set the device to CUDA if available
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+"
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("mps")




l_rate = 0.001
hidden_size = 1
# 1 to 60 time steps
time_steps = [60]
epochs = 2500
patience = 10
agent_type = 'strong'

best_val_loss = float('inf')
epochs_without_improvement = 0
result = []
training_losses = []
validation_losses = []

def create_training_traces(env, mode, inp):
    # Training
    env.reset(mode)
    if inp == 1:
        baseline_agent = baselineAgent(agent_type=agent_type)
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
    elif inp == 4:
        states = []
        agent = UCBAgent()
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


baseline_agent = None
agent = None

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

# def impute_missing_values(states):
#     imputed_states = []
#     for state in states:
#         mean_values = np.mean([v for v in state if v >= 0])
#         imputed_state = np.array([v if v >= 0 else mean_values for v in state])
#         imputed_states.append(imputed_state)
#     return np.array(imputed_states)
#
# def imv(state):
#     mean_value = np.mean([v for v in state if v >= 0])
#     imputed_state = np.array([v if v >= 0 else mean_value for v in state])
#     return imputed_state

def impute_missing_values(states):
    # median impuation
    imputed_states = []
    for state in states:
        median_values = np.median([v for v in state if v >= 0])
        imputed_state = np.array([v if v >= 0 else median_values for v in state])
        imputed_states.append(imputed_state)
    return np.array(imputed_states)

def imv(state):
    median_value = np.median([v for v in state if v >= 0])
    imputed_state = np.array([v if v >= 0 else median_value for v in state])
    return imputed_state


def get_class_distribution(labels):
    # Convert one-hot encoded labels to integer labels
    int_labels = np.argmax(labels, axis=1)
    return np.bincount(int_labels)


def upsample_classes(train_data, train_labels):
    classes = np.unique(train_labels)
    upsampled_train_data = []
    upsampled_train_labels = []

    # Get the distribution of classes
    class_distribution = get_class_distribution(train_labels)

    print("Distribution of classes before balancing:", class_distribution)

    # Set the target number of samples to be the maximum count among all classes
    target_samples = np.max(class_distribution)

    for cls in classes:
        class_idx = np.where(train_labels == cls)[0]
        class_data = train_data[class_idx]
        class_labels = train_labels[class_idx]

        if len(class_data) < target_samples:
            class_data, class_labels = resample(class_data, class_labels,
                                                replace=True,  # sample with replacement
                                                n_samples=target_samples,  # to match majority class
                                                random_state=123)  # reproducible results

        upsampled_train_data.append(class_data)
        upsampled_train_labels.append(class_labels)

    upsampled_train_labels = np.concatenate(upsampled_train_labels)

    print("Distribution of classes after balancing:", get_class_distribution(upsampled_train_labels))

    return np.concatenate(upsampled_train_data), upsampled_train_labels

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
    inp2 = int(input("1. Baseline Agent 2. D-UCB Agent: 3. SW-UCB Agent 4. UCB-1 Agent\n"))


    train_data = create_training_traces(env, 'train', inp2)
    test_data = create_training_traces(env, 'test', inp2)

    train_data = impute_missing_values(train_data)
    test_data = impute_missing_values(test_data)

    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size).to(device)
    optimizer = Adam(lstm_agent.parameters(), lr=l_rate)

    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)

    for ts in time_steps:
        # Use the function on your data
        trainX, trainY = get_XY(train_data, ts)
        trainX, trainY = upsample_classes(trainX, trainY)

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
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
            optimizer.zero_grad()
            outputs, (hidden_state, cell_state) = lstm_agent(trainX, (
            hidden_state, cell_state))
            loss = criterion(outputs, trainY)
            loss.backward()

            optimizer.step()
            # scheduler.step()

            # Validation
            val_outputs, (_, _) = lstm_agent(testX,
                                             lstm_agent.init_hidden_cell_states(
                                                 batch_size=testX.size(0)))
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
            val_loss = criterion(val_outputs, testY)
            validation_losses.append(round(val_loss.item(), 3))  # Append the reward at each timestep
            training_losses.append(round(loss.item(), 3))  # Append the reward at each timestep

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
            state = impute_missing_values([state])[0]


            if stop:
                break

        average_inference_time = statistics.mean(inference_times)

        print(f'====== RESULT ======')
        if inp2 == 1:
            print("Used Historical traces: Baseline Agent")
        if inp2 == 2:
            print("Used Historical traces: D-UCB Agent")
        if inp2 == 3:
            print("Used Historical traces: SW-UCB Agent")
        if inp2 == 4:
            print("Used Historical traces: UCB-1 Agent")

        print('[total reward]:', env.get_total_reward())
        print('[Hyperparameters]')
        print(
            "epochs: {} lr: {} \nhidden_size: {} time_steps: {} loss function: {}".format(
                epochs, l_rate, hidden_size, ts, inp1))

        total_reward = env.get_total_reward()

        # used historical trace, total reward, epochs, l_rate, hidden_size, amount of timesteps, 1: MSE, 2: MAE, 3: Huber
        result.append(
            [inp2, total_reward, best_epoch, epochs, l_rate, hidden_size, ts,
             inp1, average_inference_time])

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



"""
====== TESTING======
[total reward]: 0.559

Difference Strong Baseline = Run 3 - Baseline = 0.559 - 0.506 = 0.053
Percentage growth = (Difference / Baseline) x 100 = 0.053 / 0.506 x 100 = 10.5%

Difference Weak Baseline = Run 3 - Baseline = 0.559 - 0.317 = 0.242
Percentage growth = (Difference / Baseline) x 100 = 0.242 / 0.317 x 100 = 76.2%

[total reward]: 0.509

Difference Strong Baseline = Run 3 - Baseline = 0.509 - 0.506 = 0.003
Percentage growth = (Difference / Baseline) x 100 = 0.003 / 0.506 x 100 = 0.6%

Difference Weak Baseline = Run 3 - Baseline = 0.509 - 0.317 = 0.192
Percentage growth = (Difference / Baseline) x 100 = 0.192 / 0.317 x 100 = 60.6%

[total reward]: 0.502

Difference Strong Baseline = Run 3 - Baseline = 0.502 - 0.506 = -0.004
Percentage growth = (Difference / Baseline) x 100 = -0.004 / 0.506 x 100 = -0.8%

Difference Weak Baseline = Run 3 - Baseline = 0.502 - 0.317 = 0.185
Percentage growth = (Difference / Baseline) x 100 = 0.185 / 0.317 x 100 = 58.4%

[total reward]: 0.525

Difference Strong Baseline = Run 3 - Baseline = 0.525 - 0.506 = 0.019
Percentage growth = (Difference / Baseline) x 100 = 0.019 / 0.506 x 100 = 3.8%

Difference Weak Baseline = Run 3 - Baseline = 0.525 - 0.317 = 0.208
Percentage growth = (Difference / Baseline) x 100 = 0.208 / 0.317 x 100 = 65.6%
"""