import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import SlidingWindowUCBAgent, UCBAgent, DiscountedUCBAgent

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
        agent = SlidingWindowUCBAgent(c=3, window_size=window_size * 60)

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
        print(f'=== TRAINING ===')
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
    print(f'====== TESTING window size ======')
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

def DiscountedUCBExperiment():
    np.random.seed(0)
    env = ROFARS_v1()
    min_gamma = 0.5
    max_gamma = 1.0
    gamma_step = 0.025
    best_gamma = min_gamma
    best_reward = -np.inf

    gammas = []
    total_rewards = []

    # Find the best gamma in the training session
    for gamma in np.arange(min_gamma, max_gamma, gamma_step):
        agent = DiscountedUCBAgent(c=5, gamma=gamma)
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

        # Record the gamma and its total reward
        gammas.append(gamma)
        total_rewards.append(total_reward)

    # Use the best gamma for testing
    agent = DiscountedUCBAgent(c=5, gamma=best_gamma)
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
    print(f'====== TESTING gamma ======')
    print('[total reward]:', test_total_reward)
    print(f'Best gamma: {best_gamma}')
    print(f'Best [total reward]: {best_reward}')

    # Plot the gamma and its total reward
    plt.plot(gammas, total_rewards,
             label=f"Best gamma: {best_gamma}, Total reward: {best_reward:.3f}")
    plt.xlabel('Gamma', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Discounted UCB: Gamma vs Total Reward', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig('DiscountedUCB.png')
    plt.show()


def SWUCBOpt(agent_type):
    if agent_type == 1:
        print("UCB-1")
    elif agent_type == 2:
        print("SW-UCB")
    elif agent_type == 3:
        print("D-UCB")

    np.random.seed(0)
    env = ROFARS_v1()
    best_window_size = 50 * 60

    """TRAINING"""
    if agent_type == 1:
        agent = UCBAgent(c=3)
    elif agent_type == 2:
        agent = SlidingWindowUCBAgent(c=3, window_size=best_window_size)
    elif agent_type == 3:
        agent = DiscountedUCBAgent(c=3, gamma=0.5)
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
    print(f'=== TRAINING===')
    print('[total reward]:', total_reward)

    """TESTING"""
    if agent_type == 1:
        agent = UCBAgent(c=3)
    elif agent_type == 2:
        agent = SlidingWindowUCBAgent(c=3, window_size=best_window_size)
    elif agent_type == 3:
        agent = DiscountedUCBAgent(c=3, gamma=0.5)
    agent.initialize(env.n_camera)

    env.reset(mode='test')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        if stop:
            break

    print(f'====== TESTING======')
    print('[total reward]:', env.get_total_reward())



if __name__ == '__main__':
    print("Enter the agent you want to test: ")
    inp = int(input('1. UCB-1 \n2. SW-UCB \n3. D-UCB'))
    if inp == 1:
        SWUCBOpt(1)
    elif inp == 2:
        SWUCBOpt(2)
    elif inp == 3:
        inp2 = int(input('Find optimal gamma? (1. Yes, 2. No)'))
        if inp2 == 1:
            DiscountedUCBExperiment()
        elif inp2 == 2:
            SWUCBOpt(3)





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


Run 2 UCB1:
[total reward]: 0.449                         
====== TESTING======
[total reward]: 0.39

Difference Strong Baseline = Run 2 - Baseline = 0.39 - 0.506 = -0.116
Percentage growth = (Difference / Baseline) x 100 = -0.116 / 0.506 x 100 = -22.9%
Growth: -22.9%.

Difference Weak Baseline = Run 2 - Baseline = 0.39 - 0.317 = 0.073
Percentage growth = (Difference / Baseline) x 100 = 0.073 / 0.317 x 100 = 23.0%
Growth: 23.0%.

Run 3 D-UCB:
====== TESTING gamma ======
[total reward]: 0.377
Best gamma: 0.9750000000000004

Difference Strong Baseline = Run 3 - Baseline = 0.377 - 0.506 = -0.129
Percentage growth = (Difference / Baseline) x 100 = -0.129 / 0.506 x 100 = -25.5%
Growth: -25.5%.

Difference Weak Baseline = Run 3 - Baseline = 0.377 - 0.317 = 0.06
Percentage growth = (Difference / Baseline) x 100 = 0.06 / 0.317 x 100 = 18.9%
Growth: 18.9%.
"""


