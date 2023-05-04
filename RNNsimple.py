import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import baselineAgent, LSTM_Agent, DiscountedUCBAgent
import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

batch_size = 32
baseline_agent = None
agent = None

time_reward_agent = {}

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

            time_reward_agent[t] = reward

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

            time_reward_agent[t] = reward

            if stop:
                break

        return states

def plot_heatmap(data, title):
    plt.figure()
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    inp = int(input("1. MSE\n2. MAE \n3. Huber\n"))
    if inp == 1:
        criterion = nn.MSELoss()
    if inp == 2:
        criterion = nn.L1Loss()
    if inp == 3:
        criterion = nn.SmoothL1Loss()

    np.random.seed(0)

    env = ROFARS_v1()

    input_size = env.n_camera
    output_size = env.n_camera
    inp = int(input("1. Baseline Agent 2. UCB Agent: "))
    hidden_size = 32
    time_steps = 9*60
    epochs = 5

    train_data = create_training_traces(env, 'train', inp)
    test_data = create_training_traces(env, 'test', inp)

    train_data = impute_missing_values(train_data)
    test_data = impute_missing_values(test_data)
    criterion = nn.SmoothL1Loss()




    lstm_agent = LSTM_Agent(input_size, hidden_size, output_size)
    optimizer = Adam(lstm_agent.parameters(), lr=0.001)

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

    # Testing loop
    print('Testing LSTM Agent')
    env.reset(mode='test')
    reward_threshold = 0.1  # Adjust the threshold value as needed
    # give random scores as the initial action
    init_action = np.random.rand(env.n_camera)
    reward, state, stop = env.step(init_action)

    # Initialize the hidden and cell states for the LSTM agent
    hidden_state, cell_state = lstm_agent.init_hidden_cell_states(batch_size=1)

    time_reward_lstm = {}

    for t in tqdm(range(env.length), initial=2):
        # Prepare the input state for the LSTM agent
        input_state = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)  # Add the batch and sequence dimensions

        # Get the action from the LSTM agent, passing the hidden and cell states
        action, (hidden_state, cell_state) = lstm_agent(input_state, (
        hidden_state, cell_state))
        action = action.squeeze().detach().numpy()

        # Perform the action in the environment
        reward, state, stop = env.step(action)
        state = imv(state)

        # Visualize input states, LSTM agent's actions, and Baseline/UCB agent's actions for low rewards
        if reward < reward_threshold:
            time_reward_lstm[t] = reward


        if stop:
            break

    print(f'====== RESULT ======')
    print('[total reward]:', env.get_total_reward())

    # Filter out time steps with rewards above reward_threshold
    filtered_time_reward_lstm = {k: v for k, v in time_reward_lstm.items() if
                                 v < reward_threshold}

    # Compute the difference between LSTM and UCB/Baseline agent rewards for each time step
    differences = {}
    for t in filtered_time_reward_lstm.keys():
        lstm_reward = filtered_time_reward_lstm[t]
        if t in time_reward_agent:
            agent_reward = time_reward_agent[t]
            differences[t] = abs(lstm_reward - agent_reward)

    # Sort the differences by magnitude and take the top 10
    sorted_differences = sorted(differences.items(), key=lambda x: x[1],
                                reverse=True)[:10]

    # Print the top 10 differences
    print('Top 10 Time Steps with Highest Reward Differences')
    for t, diff in sorted_differences:
        lstm_reward = filtered_time_reward_lstm[t]
        agent_reward = time_reward_agent[t]
        print(
            f'Time Step {t}: LSTM Reward={lstm_reward:.2f}, Agent Reward={agent_reward:.2f}, Difference={diff:.2f}')
