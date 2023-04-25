# import matplotlib.pyplot as plt
# import numpy as np
#
# def mean_approach(data):
#     return np.mean(data)
#
# def sliding_window_approach(data, window_size=5):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='same')
#
# def discounted_ucb_approach(data, discount_factor=0.9):
#     ucb_values = np.zeros_like(data, dtype=float)
#     for t in range(1, len(data)):
#         ucb_values[t] = data[t] + np.sqrt((1 - discount_factor) * np.log(t) / (t * discount_factor ** t))
#     return ucb_values
#
# # Example data (replace with your own data)
# timesteps = np.arange(0, 24 * 4, 1)
# days = 4
#
# # Repeat the people_count data for 4 days
# people_count = np.array([0, 0, 3, 0, 10, 5, 10, 20, 50, 100, 400, 300, 500, 600, 500, 300, 200, 500, 0, 0, 3, 0, 4, 0])
# people_count = np.tile(people_count, days)
#
# mean_line = mean_approach(people_count)
# sliding_window_line = sliding_window_approach(people_count, window_size=5)
# discounted_ucb_line = discounted_ucb_approach(people_count, discount_factor=0.9)
#
# plt.figure(figsize=(15, 6))
# plt.plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
# plt.plot(timesteps, mean_line * np.ones_like(timesteps), linewidth=2, label='Non-Stationary: Mean Approach')
# plt.plot(timesteps, sliding_window_line, linewidth=2, label='Non-Stationary: Sliding Window Approach')
# plt.plot(timesteps, discounted_ucb_line, linewidth=2, label='Non-Stationary: Discounted Approach')
#
# plt.title("Number of People in Shopping Mall for 4 Days")
# plt.xlabel("Time of the Day (Hour)")
# plt.ylabel("Number of Faces")
#
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.savefig("presentation.png")
# plt.show()

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

fig, axs = plt.subplots(2, 1, figsize=(15, 18), sharex=True)

axs[0].plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
axs[0].plot(timesteps, mean_line * np.ones_like(timesteps), linewidth=2, label='Mean Approach')
axs[0].set_title("Stationary: Mean Approach (UCB-1)")
axs[0].set_ylabel("Number of Faces")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
axs[1].plot(timesteps, sliding_window_line, linewidth=2, label='Sliding Window Approach')
axs[1].set_title("Non-Stationary: Sliding Window Approach (SW-UCB)")
axs[1].set_ylabel("Number of Faces")
axs[1].legend()
axs[1].grid(True)

# axs[2].plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
# axs[2].plot(timesteps, discounted_ucb_line, linewidth=2, label='Discounted UCB Approach')
# axs[2].set_title("Non-Stationary: Discounted UCB Approach (D-UCB)")
# axs[2].set_xlabel("Time of the Day (Hour)")
# axs[2].set_ylabel("Number of Faces")
# axs[2].legend()
# axs[2].grid(True)

plt.tight_layout()
plt.savefig("presentation.png")
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# def mean_approach(data):
#     return np.mean(data)
#
# def sliding_window_approach(data, window_size=5):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='same')
#
# def discounted_ucb_approach(data, discount_factor=0.9):
#     ucb_values = np.zeros_like(data, dtype=float)
#     for t in range(1, len(data)):
#         ucb_values[t] = data[t] + np.sqrt((1 - discount_factor) * np.log(t) / (t * discount_factor ** t))
#     return ucb_values
#
# # Example data (replace with your own data)
# timesteps = np.arange(0, 24 * 4, 1)
# days = 4
#
# # Repeat the people_count data for 4 days
# people_count = np.array([0, 0, 3, 0, 10, 5, 10, 20, 50, 100, 400, 300, 500, 600, 500, 300, 200, 500, 0, 0, 3, 0, 4, 0])
# people_count = np.tile(people_count, days)
#
# mean_line = mean_approach(people_count)
# sliding_window_line = sliding_window_approach(people_count, window_size=5)
# discounted_ucb_line = discounted_ucb_approach(people_count, discount_factor=0.9)
#
# # Mean Approach
# plt.figure(figsize=(15, 6))
# plt.plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
# plt.plot(timesteps, mean_line * np.ones_like(timesteps), linewidth=2, label='Mean Approach')
# plt.title("Stationary: Mean Approach (UCB-1)")
# plt.xlabel("Time of the Day (Hour)")
# plt.ylabel("N Cameras")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("mean_approach.png")
# plt.show()
#
# # Sliding Window Approach
# plt.figure(figsize=(15, 6))
# plt.plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
# plt.plot(timesteps, sliding_window_line, linewidth=2, label='Sliding Window Approach')
# plt.title("Non-Stationary: Sliding Window Approach (SW-UCB)")
# plt.xlabel("Time of the Day (Hour)")
# plt.ylabel("N Cameras")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("sliding_window_approach.png")
# plt.show()
#
# # Discounted UCB Approach
# plt.figure(figsize=(15, 6))
# plt.plot(timesteps, people_count, marker='o', linewidth=2, label='Original Data')
# plt.plot(timesteps, discounted_ucb_line, linewidth=2, label='Discounted UCB Approach')
# plt.title("Non-Stationary: Discounted UCB Approach (D-UCB)")
# plt.xlabel("Time of the Day (Hour)")
# plt.ylabel("N Cameras")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("discounted_ucb_approach.png")
# plt.show()
