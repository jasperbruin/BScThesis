import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from tensorflow.keras.optimizers import Adam
from agents import baselineAgent, LSTM_Agent

np.random.seed(0)

env = ROFARS_v1()
best_total_reward = -np.inf

input_size = env.n_camera
output_size = env.n_camera

# Grid search parameters
learning_rates = [0.0001, 0.001, 0.01]
epochs_list = [50, 100, 150]
hidden_sizes = [16, 32, 64]

results = []

for lr in learning_rates:
    for epochs in epochs_list:
        for hidden_size in hidden_sizes:
            lstm_agent = LSTM_Agent(input_size, hidden_size, output_size)

            optimizer = Adam(lr=lr)
            lstm_agent.compile(optimizer, loss='mse')

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
            lstm_agent.fit(states, actions, epochs=epochs, verbose=1)

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

            # Store results
            results.append((lr, epochs, hidden_size, total_reward))

            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_params = (lr, epochs, hidden_size)

print(f'Optimal Parameters: Learning Rate = {best_params[0]}, Epochs = {best_params[1]}, Hidden Size = {best_params[2]}')

# Plot the results
fig, ax = plt.subplots()
x = np.arange(len(results))
y = [result[-1] for result in results]
ax.plot(x, y, marker='o')

ax.set_xlabel('Parameter Combination')
ax.set_ylabel('Total Reward')
ax.set_title('Optimal Parameters Search')

plt.show()
