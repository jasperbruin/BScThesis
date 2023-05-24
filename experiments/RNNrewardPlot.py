import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv('/results.csv', header=None)

# Set the column names
df.columns = ['used historical trace', 'total reward', 'epochs', 'l_rate', 'hidden_size', 'amount of timesteps', 'loss']

# Map the used historical trace and loss to their names
df['used historical trace'] = df['used historical trace'].map({1: 'Baseline Agent', 2: 'D-UCB', 3: 'SW-UCB'})
df['loss'] = df['loss'].map({1: 'MSE', 2: 'MAE', 3: 'Huber'})

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Learning rate vs total reward
for name, group in df.groupby('loss'):
    axs[0,0].scatter(group['l_rate'], group['total reward'], label=name)
axs[0,0].legend(title='Loss')
axs[0,0].set_xlabel('Learning Rate')
axs[0,0].set_ylabel('Total Reward')
axs[0,0].set_title('Learning rate vs Total Reward')

# Plot 2: Hidden size vs total reward
for name, group in df.groupby('loss'):
    axs[0,1].scatter(group['hidden_size'], group['total reward'], label=name)
axs[0,1].legend(title='Loss')
axs[0,1].set_xlabel('Hidden Size')
axs[0,1].set_ylabel('Total Reward')
axs[0,1].set_title('Hidden Size vs Total Reward')

# Plot 3: Amount of timesteps vs total reward
for name, group in df.groupby('loss'):
    axs[1,0].scatter(group['amount of timesteps'], group['total reward'], label=name)
axs[1,0].legend(title='Loss')
axs[1,0].set_xlabel('Amount of Timesteps')
axs[1,0].set_ylabel('Total Reward')
axs[1,0].set_title('Amount of Timesteps vs Total Reward')

# Plot 4: Used historical trace vs total reward
for name, group in df.groupby('loss'):
    axs[1,1].scatter(group['used historical trace'], group['total reward'], label=name)
axs[1,1].legend(title='Loss')
axs[1,1].set_xlabel('Used Historical Trace')
axs[1,1].set_ylabel('Total Reward')
axs[1,1].set_title('Used Historical Trace vs Total Reward')

# Show the plot
plt.show()
