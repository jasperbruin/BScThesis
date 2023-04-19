import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import SlidingWindowUCBAgent

def SWUCBExperiment():
    np.random.seed(0)
    env = ROFARS_v1()
    max_window_size = 100
    best_window_size = 1
    best_reward = -np.inf

    window_sizes = []
    total_rewards = []

    # Find the best sliding window in the training session
    for window_size in range(1, max_window_size + 1):
        agent = SlidingWindowUCBAgent(c=3, window_size=window_size * 60,
                                      mode='sw')  # Sliding Window mode

        agent.initialize(env.n_camera)

        # Training loop
        env.reset(mode='train')

        for t in tqdm(range(env.length), initial=2):
            action = agent.get_action()
            reward, state, stop = env.step(action)

            # Update the UCB Agent
            agent.update(action, state)

            if stop:
                break

        total_reward = env.get_total_reward()
        print(f'=== TRAINING window size {window_size} ===')
        print('[total reward]:', total_reward)

        # Save the best window size and total reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_window_size = window_size

        # Record the window size and its total reward
        window_sizes.append(window_size)
        total_rewards.append(total_reward)

    # Use the best sliding window for testing
    agent = SlidingWindowUCBAgent(c=3, window_size=best_window_size)
    agent.initialize(env.n_camera)

    env.reset(mode='test')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        if stop:
            break

    test_total_reward = env.get_total_reward()
    print(f'====== TESTING window size {best_window_size} ======')
    print('[total reward]:', test_total_reward)
    print(f'Best window size: {best_window_size}')
    print(f'Best [total reward]: {best_reward}')

    # Plot the window size and its total reward
    plt.plot(window_sizes, total_rewards,
             label=f"Best window size: {best_window_size}, Total reward: {best_reward:.3f}")
    plt.xlabel('Window Size', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Sliding Window UCB: Window Size vs Total Reward', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig('UCB.png')
    plt.show()


def SWUCBOpt():
    np.random.seed(0)
    env = ROFARS_v1()
    best_window_size = 50 * 60

    """TRAINING: Best window size is 50 * 60 minutes"""
    agent = SlidingWindowUCBAgent(c=3, window_size=best_window_size, mode='sw')
    agent.initialize(env.n_camera)

    # Training loop
    env.reset(mode='train')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        if stop:
            break

    total_reward = env.get_total_reward()
    print(f'=== TRAINING window size 50 * 60 ===')
    print('[total reward]:', total_reward)

    """TESTING: Best window size is 50 * 60 minutes"""
    agent = SlidingWindowUCBAgent(c=3, window_size=best_window_size)
    agent.initialize(env.n_camera)

    env.reset(mode='test')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        if stop:
            break

    print(f'====== TESTING window size 50 * 60 ======')
    print('[total reward]:', env.get_total_reward())

def DUCBExperiment():
    np.random.seed(0)
    env = ROFARS_v1()

    gamma_values = np.linspace(0.5, 0.99, 20)  # Adjust the range and number of gamma values as needed
    best_gamma = 0.99
    best_reward = -np.inf

    gamma_rewards = []

    # Find the best gamma in the training session
    for gamma in gamma_values:
        agent = SlidingWindowUCBAgent(c=3, window_size=1, mode='d', gamma=gamma)
        agent.initialize(env.n_camera)

        # Training loop
        env.reset(mode='train')

        for t in tqdm(range(env.length), initial=2):
            action = agent.get_action()
            reward, state, stop = env.step(action)

            # Update the UCB Agent
            agent.update(action, state)

            if stop:
                break

        total_reward = env.get_total_reward()
        print(f'=== TRAINING gamma {gamma} ===')
        print('[total reward]:', total_reward)

        # Save the best gamma and total reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_gamma = gamma

        # Record the gamma value and its total reward
        gamma_rewards.append(total_reward)

    # Use the best gamma for testing
    agent = SlidingWindowUCBAgent(c=3, window_size=1, mode='d', gamma=best_gamma)
    agent.initialize(env.n_camera)

    env.reset(mode='test')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        if stop:
            break

    test_total_reward = env.get_total_reward()
    print(f'====== TESTING gamma {best_gamma} ======')
    print('[total reward]:', test_total_reward)
    print(f'Best gamma: {best_gamma}')
    print(f'Best [total reward]: {best_reward}')

    # Plot the gamma values and their total rewards
    plt.plot(gamma_values, gamma_rewards, label=f"Best gamma: {best_gamma:.2f}, Total reward: {best_reward:.2f}")
    plt.xlabel('Gamma Value', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Discounted UCB: Gamma Value vs Total Reward', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig('DUCB.png')
    plt.show()


def BothExperiment():
    np.random.seed(0)
    env = ROFARS_v1()

    gamma_values = np.linspace(0.5, 0.99, 20)  # Adjust the range and number of gamma values as needed
    best_gamma = 0.99
    best_reward = -np.inf

    gamma_rewards = []

    # Find the best gamma in the training session
    for gamma in gamma_values:
        agent = SlidingWindowUCBAgent(c=3, window_size=50 * 60, mode='both', gamma=gamma)
        agent.initialize(env.n_camera)

        # Training loop
        env.reset(mode='train')

        for t in tqdm(range(env.length), initial=2):
            action = agent.get_action()
            reward, state, stop = env.step(action)

            # Update the UCB Agent
            agent.update(action, state)

            if stop:
                break

        total_reward = env.get_total_reward()
        print(f'=== TRAINING gamma {gamma} ===')
        print('[total reward]:', total_reward)

        # Save the best gamma and total reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_gamma = gamma

        # Record the gamma value and its total reward
        gamma_rewards.append(total_reward)

    # Use the best gamma for testing
    agent = SlidingWindowUCBAgent(c=3, window_size=50 * 60, mode='d', gamma=best_gamma)
    agent.initialize(env.n_camera)

    env.reset(mode='test')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        if stop:
            break

    test_total_reward = env.get_total_reward()
    print(f'====== TESTING gamma {best_gamma} ======')
    print('[total reward]:', test_total_reward)
    print(f'Best gamma: {best_gamma}')
    print(f'Best [total reward]: {best_reward}')

    # Plot the gamma values and their total rewards
    plt.plot(gamma_values, gamma_rewards, label=f"Best gamma: {best_gamma:.2f}, Total reward: {best_reward:.2f}")
    plt.xlabel('Gamma Value', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Discounted UCB: Gamma Value vs Total Reward', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig('DUCB.png')
    plt.show()

if __name__ == '__main__':
    inp = input(
        'Choose an experiment:\n1. Sliding Window UCB\n2. Discounted UCB\n3. Both\n')
    if inp == '1':
        inp2 = input('Choose an experiment:\n1. Find optimal sliding window \n2. Use optimal sliding window (50 * 60) minutes\n')
        if inp2 == '1':
            SWUCBExperiment()
        if inp2 == '2':
            SWUCBOpt()
    elif inp == '2':
        DUCBExperiment()
    elif inp == '3':
        BothExperiment()
    else:
        print("Invalid input")


"""
Baseline:
====== TESTING ======
[total reward]: 0.506


Run 1 SW-UCB:
TRAINING window size 50 * 60 ===
[total reward]: 0.558                       
====== TESTING window size 50 * 60 ======
[total reward]: 0.528

Difference Strong Baseline = Run 1 - Baseline = 0.528 - 0.506 = 0.022
Percentage growth = (Difference / Baseline) x 100 = 0.022 / 0.506 x 100 = 4.3%
Growth: 4.3%.

Difference Weak Baseline = Run 1 - Baseline = 0.528 - 0.317 = 0.211
Percentage growth = (Difference / Baseline) x 100 = 0.211 / 0.317 x 100 = 66.4%
Growth: 66.4%. 


Run 2: SW-D:
I note that the environment's rewards are not sensitive to the discount factor 
or if the agent is not exploring effectively, the total reward might still remain 
the same despite the change. 
"""


