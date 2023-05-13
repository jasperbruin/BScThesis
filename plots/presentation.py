import matplotlib.pyplot as plt
import numpy as np

def mean_approach(data):
    return np.mean(data)

def sliding_window_approach(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def discounted_ucb_approach(data, discount_factor=0.9):
    ucb_values = np.zeros_like(data, dtype=float)
    for t in range(1, len(data)):
        ucb_values[t] = data[t] + np.sqrt((1 - discount_factor) * np.log(t) / (t * discount_factor ** t))
    return ucb_values

# Example data (replace with your own data)
timesteps = np.arange(0, 24 * 4, 1)
days = 4

# Repeat the people_count data for 4 days
people_count = np.array([0, 0, 3, 0, 10, 5, 10, 20, 50, 100, 400, 300, 500, 600, 500, 300, 200, 500, 0, 0, 3, 0, 4, 0])
people_count = np.tile(people_count, days)

mean_line = mean_approach(people_count)
sliding_window_line = sliding_window_approach(people_count, window_size=5)
discounted_ucb_line = discounted_ucb_approach(people_count, discount_factor=0.9)

# Create the first figure
fig1, ax1 = plt.subplots(figsize=(15, 6))
ax1.plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
ax1.plot(timesteps, mean_line * np.ones_like(timesteps), linewidth=2, label='Mean Approach')
ax1.set_title("Stationary: Mean Approach (UCB-1)")
ax1.set_ylabel("Number of Faces")
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig("mean_approach_plot.png")

# Create the second figure
fig2, ax2 = plt.subplots(figsize=(15, 6))
ax2.plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
ax2.plot(timesteps, sliding_window_line, linewidth=2, label='Sliding Window Approach')
ax2.set_title("Non-Stationary: Sliding Window Approach (SW-UCB)")
ax2.set_ylabel("Number of Faces")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("sliding_window_plot.png")

plt.show()
