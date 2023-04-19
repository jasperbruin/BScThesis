import matplotlib.pyplot as plt

def add_text_label(ax, x, y, value, color='white', fontweight='bold'):
    ax.text(x, y, f'{value:.1f}%', ha='center', va='center', color=color, fontweight=fontweight)

# Data
baseline_strong = 0.506
baseline_weak = 0.317
runs = ['Run 1 SW-UCB', 'Run 2 UCB1', 'Run 3 D-UCB']
test_rewards = [0.528, 0.39, 0.377]

# Calculate percentage growths
growth_strong = [(reward - baseline_strong) / baseline_strong * 100 for reward in test_rewards]
growth_weak = [(reward - baseline_weak) / baseline_weak * 100 for reward in test_rewards]

# Plot
fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

bar1 = ax.bar(runs, growth_strong, width=bar_width, alpha=opacity, label='Strong Baseline')
bar2 = ax.bar(runs, growth_weak, width=bar_width, bottom=growth_strong, alpha=opacity, label='Weak Baseline')

ax.set_ylabel('Percentage Growth (%)')
ax.set_title('Percentage Growth for Each Run Compared to Baselines')
ax.legend()

for i, (strong_growth, weak_growth) in enumerate(zip(growth_strong, growth_weak)):
    add_text_label(ax, i - bar_width / 2, strong_growth / 2, strong_growth)
    add_text_label(ax, i - bar_width / 2, strong_growth + weak_growth / 2, weak_growth)

plt.tight_layout()
plt.show()
