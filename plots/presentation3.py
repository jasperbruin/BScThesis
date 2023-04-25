import matplotlib.pyplot as plt
import numpy as np

def sliding_window_average(data, tau=5):
    return np.convolve(data, np.ones(tau)/tau, mode='same')

def padding_function(data, tau, B=1, xi=2):
    t = len(data)
    N_t = np.convolve(np.ones_like(data), np.ones(tau), mode='same')
    c_t = B * np.sqrt((xi * np.log(np.minimum(t, tau))) / N_t)
    return c_t

# Example data (replace with your own data)
np.random.seed(42)
data = np.random.normal(100, 20, size=50)

tau = 5
window_average = sliding_window_average(data, tau)
padding = padding_function(data, tau)

plt.figure(figsize=(10, 6))
plt.plot(data, marker='o', linestyle='-', label='Original Data')
plt.plot(window_average, linestyle='-', color='orange', label='Sliding Window Average')
plt.plot(window_average + padding, linestyle='--', color='green', label='Padding Upper Bound')
plt.plot(window_average - padding, linestyle='--', color='green', label='Padding Lower Bound')

plt.title("Sliding Window UCB Padding Function")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("sliding_window_ucb_padding.png")
plt.show()
