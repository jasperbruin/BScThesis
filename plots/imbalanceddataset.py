import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# Define mean and covariance for each group
mean_red_imbalanced = [2, 2]
cov_red_imbalanced = [[1, 0], [0, 1]]

mean_blue_imbalanced = [-2, -2]
cov_blue_imbalanced = [[1, 0], [0, 1]]

# Generate imbalanced data
red_data_imbalanced = np.random.multivariate_normal(mean_red_imbalanced, cov_red_imbalanced, 4)
blue_data_imbalanced = np.random.multivariate_normal(mean_blue_imbalanced, cov_blue_imbalanced, 50)

# Define mean and covariance for balanced data
mean_red_balanced = [2, 2]
cov_red_balanced = [[1, 0], [0, 1]]

mean_blue_balanced = [-2, -2]
cov_blue_balanced = [[1, 0], [0, 1]]

# Generate balanced data
red_data_balanced = np.random.multivariate_normal(mean_red_balanced, cov_red_balanced, 27)
blue_data_balanced = np.random.multivariate_normal(mean_blue_balanced, cov_blue_balanced, 27)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

# Plot imbalanced data
axes[0].scatter(red_data_imbalanced[:, 0], red_data_imbalanced[:, 1], color='red', label='class $y=1$')
axes[0].scatter(blue_data_imbalanced[:, 0], blue_data_imbalanced[:, 1], color='blue', label='class $y=0$')
axes[0].set_title('Imbalanced Dataset')
axes[0].legend()

# Plot balanced data
axes[1].scatter(red_data_balanced[:, 0], red_data_balanced[:, 1], color='red', label='class $y=1$')
axes[1].scatter(blue_data_balanced[:, 0], blue_data_balanced[:, 1], color='blue', label='class $y=0$')
axes[1].set_title('Balanced Dataset')
axes[1].legend()

plt.tight_layout()
plt.savefig('imbalanceddataset.png')
plt.show()
