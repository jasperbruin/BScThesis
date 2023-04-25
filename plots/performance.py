import pandas as pd
import matplotlib.pyplot as plt

data = {'Run': ['Baseline', 'Run 1 SW-UCB', 'Run 2 UCB1', 'Run 3 D-UCB'],
        'Total Reward': [0.506, 0.524, 0.499, 0.559],
        'Difference Strong Baseline': [0, 0.018, -0.007, 0.053],
        'Percentage Growth Strong Baseline': [0, 3.6, -1.4, 10.5],
        'Difference Weak Baseline': [0, 0.207, 0.182, 0.242],
        'Percentage Growth Weak Baseline': [0, 65.3, 57.4, 76.2]}

df = pd.DataFrame(data)

# Set the Run column as the index
df.set_index('Run', inplace=True)

# Format the table using style
styled_table = df.style.format({
    'Total Reward': '{:.3f}',
    'Difference Strong Baseline': '{:.3f}',
    'Percentage Growth Strong Baseline': '{:.1f}%',
    'Difference Weak Baseline': '{:.3f}',
    'Percentage Growth Weak Baseline': '{:.1f}%'
})

# Save the table as a figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc='center', loc='center')
plt.savefig('table_figure.png', dpi=100, bbox_inches='tight')
plt.show()
