import matplotlib.pyplot as plt
import numpy as np

# Names of agents
agents = ["Simple Baseline", "Strong Baseline", "SW-UCB", "UCB-1", "D-UCB", "LSTM Agent: Strong Baseline",
          "LSTM Agent: Simple Baseline", "LSTM Agent: UCB-1", "LSTM Agent: SW-UCB", "LSTM Agent: D-UCB"]

# Corresponding average results
average_results = [0.317, 0.506, 0.562, 0.499, 0.599, 0.534, 0.333, 0.41, 0.525, 0.512]

# Sort agents and average_results by average_results
sort_index = np.argsort(average_results)[::-1]  # Reverse order
agents = [agents[i] for i in sort_index]
average_results = [average_results[i] for i in sort_index]

y_pos = np.arange(len(agents))

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Set colors for the bars
colors = ['gray' if agent != 'Strong Baseline' and agent != 'Simple Baseline' else 'green' for agent in agents]

# Create horizontal bars with colors
bars = ax.barh(y_pos, average_results, align='center', color=colors)

# Label the bars with the average result value
for i in range(len(bars)):
    ax.text(bars[i].get_width() + 0.01, bars[i].get_y() + bars[i].get_height()/2,
            f'{average_results[i]:.3f}', color='black', va='center')

# Set a buffer space for x-axis
ax.set_xlim([0, max(average_results) + 0.1])

# Invert the y-axis to have the highest value at the top
ax.invert_yaxis()

# Set title and labels for axes
ax.set_xlabel("Average Result", fontsize=12)
ax.set_title("Performance of Agents", fontsize=14)

# Place y labels (agent names) inside the bars
ax.set_yticks(y_pos)
for i in range(len(agents)):
    ax.text(0.01, i, agents[i], va='center', ha='left', color='white', fontsize=10)


plt.savefig('averageReward.png', dpi=500)
plt.show()
