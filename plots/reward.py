import matplotlib.pyplot as plt

# Algorithm names and corresponding total rewards
algorithms = ['Simple Baseline', 'Strong Baseline', 'UCB-1', 'SW-UCB', 'D-UCB']
rewards = [0.317, 0.506, 0.499, 0.589, 0.559]

# Create a bar plot
fig, ax = plt.subplots()
ax.bar(algorithms, rewards)

# Add labels and title
ax.set_xlabel('Algorithms', labelpad=-20)
ax.set_ylabel('Total Reward')
ax.set_title('Total Reward by Algorithm')

# Rotate x-axis labels
plt.xticks(rotation=30)

# Display the plot and save with adjusted bounding box
plt.savefig('reward.png', bbox_inches='tight')
plt.show()
