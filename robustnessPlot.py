# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def plot_robustness_test(file_name, plot_title, output_file):
#     # load data from csv
#     df = pd.read_csv(file_name)
#
#     # unpack data
#     budget_ratios = df['budget_ratios']
#     rewards_ucb1 = df['rewards_ucb1']
#     rewards_sw_ucb = df['rewards_sw_ucb']
#     rewards_d_ucb = df['rewards_d_ucb']
#     rewards_simple_baseline = df['rewards_simple_baseline']
#     rewards_strong_baseline = df['rewards_strong_baseline']
#
#     # Plot the results
#     plt.figure(figsize=(8, 6))
#     plt.plot(budget_ratios, rewards_ucb1, marker='o', markersize=8,
#              linestyle='-', linewidth=2, label="UCB-1")
#     plt.plot(budget_ratios, rewards_sw_ucb, marker='s', markersize=8,
#              linestyle='--', linewidth=2, label="SW-UCB")
#     plt.plot(budget_ratios, rewards_d_ucb, marker='^', markersize=8,
#              linestyle='-.', linewidth=2, label="D-UCB")
#     plt.plot(budget_ratios, rewards_simple_baseline, marker='x', markersize=8,
#              linestyle=':', linewidth=2, label="Simple Baseline")
#     plt.plot(budget_ratios, rewards_strong_baseline, marker='D', markersize=8,
#              linestyle=(0, (5, 1)), linewidth=2, label="Strong Baseline")
#
#     plt.xlabel("Budget Ratio", fontsize=14)
#     plt.ylabel("Total Reward", fontsize=14)
#     plt.title(plot_title, fontsize=16)
#     plt.legend(fontsize=12)
#     plt.grid(alpha=0.3)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(output_file)
#     plt.show()
#
#
# if __name__ == '__main__':
#     # Plot the robustness test for UCB
#     plot_robustness_test('robustness_test.csv',
#                          'Robustness Test [Testing] for UCB-1, SW-UCB, and D-UCB',
#                          'Robustness_Test_UCB_testing.png')
#
#     # Plot the robustness test for LSTM
#     plot_robustness_test('robustness_test2.csv',
#                          'Robustness Test [Testing] for LSTM',
#                          'Robustness_Test_LSTM_testing.png')
import pandas as pd
import matplotlib.pyplot as plt


def plot_robustness_test(ax, df, title):
    # unpack data
    budget_ratios = df['budget_ratios']
    rewards_ucb1 = df['rewards_ucb1']
    rewards_sw_ucb = df['rewards_sw_ucb']
    rewards_d_ucb = df['rewards_d_ucb']
    rewards_simple_baseline = df['rewards_simple_baseline']
    rewards_strong_baseline = df['rewards_strong_baseline']

    ax.plot(budget_ratios, rewards_ucb1, marker='o', markersize=8,
            linestyle='-', linewidth=2, label="UCB-1")
    ax.plot(budget_ratios, rewards_sw_ucb, marker='s', markersize=8,
            linestyle='--', linewidth=2, label="SW-UCB")
    ax.plot(budget_ratios, rewards_d_ucb, marker='^', markersize=8,
            linestyle='-.', linewidth=2, label="D-UCB")
    ax.plot(budget_ratios, rewards_simple_baseline, marker='x', markersize=8,
            linestyle=':', linewidth=2, label="Simple Baseline")
    ax.plot(budget_ratios, rewards_strong_baseline, marker='D', markersize=8,
            linestyle=(0, (5, 1)), linewidth=2, label="Strong Baseline")

    ax.set_xlabel("Budget Ratio", fontsize=14)
    ax.set_ylabel("Total Reward", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)


if __name__ == '__main__':
    # Load data
    df1 = pd.read_csv('robustness_test.csv')
    df2 = pd.read_csv('robustness_test2.csv')

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot the robustness test for UCB
    plot_robustness_test(axs[0], df1,
                         'Robustness Test [Testing] for UCB-1, SW-UCB, and D-UCB')

    # Plot the robustness test for LSTM
    plot_robustness_test(axs[1], df2, 'Robustness Test [Testing] for LSTM')

    plt.tight_layout()
    plt.savefig("Robustness_Test_Comparison.png")
    plt.show()
