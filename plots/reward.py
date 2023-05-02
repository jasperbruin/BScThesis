import matplotlib.pyplot as plt

# Algorithm names and corresponding total rewards
algorithms = ['Simple Baseline', 'Strong Baseline', 'UCB-1', 'SW-UCB', 'D-UCB']
rewards = [0.317, 0.506, 0.499, 0.553, 0.559]

# Create a bar plot
fig, ax = plt.subplots()
ax.bar(algorithms, rewards)

# Add labels and title
ax.set_xlabel('Algorithms')
ax.set_ylabel('Total Reward')
ax.set_title('Total Reward by Algorithm')

# Display the plot
plt.savefig('reward.png')
plt.show()
