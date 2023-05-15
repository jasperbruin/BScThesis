import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import baselineAgent, LSTM_Agent, DiscountedUCBAgent, SlidingWindowUCBAgent
import torch
from torch import nn
from torch.optim import Adam
import csv

# Function to set the device to CUDA if available
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

# Setting the device
device = get_device()

baseline_agent = None
agent = None

# Hyperparameters
batch_size = 8
l_rate = 0.001
hidden_size = 16
time_steps = [9*60]
#time_steps = [9*60]
epochs = 5000
patience = 5


best_val_loss = float('inf')
epochs_without_improvement = 0
result = []
# ['Agent', 'Total Reward', 'Epochs', 'Learning Rate', 'Batch Size', 'Hidden Size', 'Time Steps', 'Loss Function']



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
            action = action.squeeze().detach().numpy()

            # Perform the action in the environment
            reward, state, stop = env.step(action)
            state = impute_missing_values([state])[0]

            if stop:
                break

        print(f'====== RESULT ======')
        if inp2 == 1:
            print("Baseline Agent")
        if inp2 == 2:
            print("D-UCB Agent")
        if inp2 == 3:
            print("SW-UCB Agent")
        print('[total reward]:', env.get_total_reward())
        print('[Hyperparameters]')
        print("epochs: {} lr: {} batch_size: {} \nhidden_size: {} time_steps: {} loss function: {}".format(epochs, l_rate, batch_size, hidden_size, ts, inp1))


        total_reward = env.get_total_reward()

        result.append([inp2, total_reward, epochs, l_rate, batch_size, hidden_size, ts, inp1])

        with open('results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in result:
                writer.writerow(row)


"""
====== RECORD ======
SW-UCB Agent: 0.562

====== RESULT ======
Baseline Agent
[total reward]: 0.501
[Hyperparameters]:  1000 0.001 64 60 64 5 1


====== RESULT ======
Baseline Agent
[total reward]: 0.516
[Hyperparameters]: 
epochs: 1000 lr: 0.001 batch_size: 64 
hidden_size: 32 time_steps: 120 loss function: 1

====== RESULT ======
Baseline Agent
[total reward]: 0.51
[Hyperparameters]
epochs: 1000 lr: 0.001 batch_size: 128 
hidden_size: 32 time_steps: 120 loss function: 1


====== RESULT ======
Baseline Agent
[total reward]: 0.501
[Hyperparameters]
epochs: 1000 lr: 0.001 batch_size: 64 
hidden_size: 32 time_steps: 180 loss function: 1


====== RESULT ======
Baseline Agent
[total reward]: 0.517
[Hyperparameters]
epochs: 1000 lr: 0.001 batch_size: 64 
hidden_size: 32 time_steps: 240 loss function: 1

====== RESULT ======
Baseline Agent
[total reward]: 0.496
[Hyperparameters]
epochs: 1000 lr: 0.001 batch_size: 64 
hidden_size: 32 time_steps: 300 loss function: 1


"""


"""
====== RESULT ======
D-UCB Agent
[total reward]: 0.504
[Hyperparameters]
epochs: 1000 lr: 0.001 batch_size: 64 
hidden_size: 32 time_steps: 120 loss function: 1

====== RESULT ======
SW-UCB Agent
[total reward]: 0.493
[Hyperparameters]
epochs: 1000 lr: 0.001 batch_size: 32 
hidden_size: 32 time_steps: 120 loss function: 1
"""

"""
====== TESTING======
[total reward]: 0.559
Best gamma: 0.999

Difference Strong Baseline = Run 3 - Baseline = 0.559 - 0.506 = 0.053
Percentage growth = (Difference / Baseline) x 100 = 0.053 / 0.506 x 100 = 10.5%

Difference Weak Baseline = Run 3 - Baseline = 0.559 - 0.317 = 0.242
Percentage growth = (Difference / Baseline) x 100 = 0.242 / 0.317 x 100 = 76.2%


====== RESULT ======
Baseline Agent
[total reward]: 0.517

Difference Strong Baseline = Run 3 - Baseline = 0.517 - 0.506 = 0.011
Percentage growth = (Difference / Baseline) x 100 = 0.011 / 0.506 x 100 = 2.2%

Difference Weak Baseline = Run 3 - Baseline = 0.517 - 0.317 = 0.2
Percentage growth = (Difference / Baseline) x 100 = 0.2 / 0.317 x 100 = 63.1%


====== RESULT ======
D-UCB Agent
[total reward]: 0.498
[Hyperparameters]
epochs: 1000 lr: 0.001 batch_size: 64 
hidden_size: 32 time_steps: 240 loss function: 1

Difference Strong Baseline = Run 3 - Baseline = 0.498 - 0.506 = -0.008
Percentage growth = (Difference / Baseline) x 100 = -0.008 / 0.506 x 100 = -1.6%

Difference Weak Baseline = Run 3 - Baseline = 0.498 - 0.317 = 0.181
Percentage growth = (Difference / Baseline) x 100 = 0.181 / 0.317 x 100 = 57.1%



====== RESULT ======
SW-UCB Agent
[total reward]: 0.499
[Hyperparameters]
epochs: 1000 lr: 0.001 batch_size: 64 
hidden_size: 32 time_steps: 240 loss function: 1

Difference Strong Baseline = Run 3 - Baseline = 0.499 - 0.506 = -0.007


"""