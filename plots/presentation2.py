import matplotlib.pyplot as plt
import numpy as np

def mean_approach(data):
    return np.mean(data)

def sliding_window_approach(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def discounted_ucb_approach(data, discount_factor=0.9, noise_std=30):
    ucb_values = np.zeros_like(data, dtype=float)
    for t in range(1, len(data)):
        ucb_values[t] = data[t] + np.sqrt((1 - discount_factor) * np.log(t) / (t * discount_factor ** t))
    ucb_values += np.random.normal(0, noise_std, size=ucb_values.shape)
    return ucb_values

# Example data (replace with your own data)
timesteps = np.arange(0, 24 * 4, 1)
days = 4

# Repeat the people_count data for 4 days
people_count = np.array([0, 0, 3, 0, 10, 5, 10, 20, 50, 100, 400, 300, 500, 600, 500, 300, 200, 500, 0, 0, 3, 0, 4, 0])
people_count = np.tile(people_count, days)

mean_line = mean_approach(people_count)
sliding_window_line = sliding_window_approach(people_count, window_size=5)
discounted_ucb_line = discounted_ucb_approach(people_count, discount_factor=0.9, noise_std=30)


# Discounted UCB Approach
plt.figure(figsize=(15, 6))
plt.plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
plt.plot(timesteps, discounted_ucb_line, linewidth=2, label='Discounted UCB Approach')
plt.title("Non-Stationary: Discounted UCB Approach (D-UCB)")
plt.xlabel("Time of the Day (Hour)")
plt.ylabel("N Cameras")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("discounted_ucb_approach_with_noise.png")
plt.show()
