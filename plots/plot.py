import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def add_text_label(ax, x, y, value, color='black', fontweight='bold', fontsize=12):
    ax.text(x, y, f'{value:.1f}%', ha='center', va='bottom', color=color, fontweight=fontweight, fontsize=fontsize)

# Data
baseline_strong = 0.506
baseline_weak = 0.317
runs = ['SW-UCB', 'UCB1', 'D-UCB']
test_rewards = [0.524, 0.514, 0.389]

# Calculate percentage growths
growth_strong = [(reward - baseline_strong) / baseline_strong * 100 for reward in test_rewards]
growth_weak = [(reward - baseline_weak) / baseline_weak * 100 for reward in test_rewards]

# Plot
fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

# Use a more visually appealing color palette
colors = sns.color_palette("muted")

# Create two separate bars for strong and weak baselines
bar1_strong = ax.bar(np.arange(len(runs))-bar_width/2, growth_strong, width=bar_width, alpha=opacity, color=colors[0], label='Strong Baseline')
bar1_weak = ax.bar(np.arange(len(runs))+bar_width/2, growth_weak, width=bar_width, alpha=opacity, color=colors[1], label='Weak Baseline')

# Increase the font size of labels and title
ax.set_ylabel('Percentage Growth (%)', fontsize=14)
ax.set_title('Percentage Growth for Each Run Compared to Baselines', fontsize=14)

# Add gridlines
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_xticks(np.arange(len(runs)))
ax.set_xticklabels(runs)

ax.legend(fontsize=12)

# Add labels to each bar
for i, (strong_growth, weak_growth) in enumerate(zip(growth_strong, growth_weak)):
    add_text_label(ax, i - bar_width / 2, strong_growth, strong_growth)
    add_text_label(ax, i + bar_width / 2, weak_growth, weak_growth)

plt.tight_layout()
plt.savefig('UCBtest.png')
plt.show()
