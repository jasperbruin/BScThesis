import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create original data
np.random.seed(0)
X = np.linspace(0, 10, 100)
Y = np.cos(-X**2/8.0) + 0.1 * np.random.randn(100)

# Introduce missing values
Y[45:55] = np.nan

# Create DataFrame
df = pd.DataFrame({'X': X, 'Y': Y})

# Interpolate missing values
df['Y_interp'] = df['Y'].interpolate(method='nearest')

# Plot original data with missing values
plt.plot(df['X'], df['Y'], color='red', label='Original data')

# Plot interpolated values
plt.plot(df['X'], df['Y_interp'], color='blue', label='Interpolated data')

# Highlight interpolated region
plt.fill_between(df['X'], df['Y_interp'], color='grey', alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.title('Nearest Neighbour Interpolation')
plt.show()
