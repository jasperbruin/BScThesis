import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import SlidingWindowUCBAgent, UCBAgent, DiscountedUCBAgent
import time

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
        agent = SlidingWindowUCBAgent(window_size=window_size * 60)

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
        print(f'=== TRAINING === window size: {window_size}')
        print('[total reward]:', total_reward)

        # Save the best window size and total reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_window_size = window_size

        # Record the window size and its total reward
        window_sizes.append(window_size)
        total_rewards.append(total_reward)

    # Use the best sliding window for testing
    agent = SlidingWindowUCBAgent(window_size=best_window_size * 60)
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
    min_gamma = 0.99
    max_gamma = 1.0
    gamma_step = 0.00025
    best_gamma = min_gamma
    best_reward = -np.inf

    gammas = []
    total_rewards = []

    # Find the best gamma in the training session
    for gamma in np.arange(min_gamma, max_gamma, gamma_step):
        agent = DiscountedUCBAgent(gamma=gamma)
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
    agent = DiscountedUCBAgent(gamma=best_gamma)
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
    best_window_size = 0
    best_gamma = 0


    """TRAINING"""
    if agent_type == 1:
        agent = UCBAgent()
    elif agent_type == 2:
        inp = int(input("Enter the window size: "))
        best_window_size = inp * 60
        agent = SlidingWindowUCBAgent(window_size=best_window_size)
    elif agent_type == 3:
        best_gamma = float(input("Enter the gamma: "))
        agent = DiscountedUCBAgent(gamma=best_gamma)
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
        agent = UCBAgent()
    elif agent_type == 2:
        agent = SlidingWindowUCBAgent(window_size=best_window_size)
    elif agent_type == 3:
        agent = DiscountedUCBAgent(gamma=best_gamma)
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


def timeexperiment(agent_type):
    if agent_type == 1:
        print("UCB-1")
    elif agent_type == 2:
        print("SW-UCB")
    elif agent_type == 3:
        print("D-UCB")

    np.random.seed(0)
    env = ROFARS_v1()

    if agent_type == 1:
        agent = UCBAgent()
    elif agent_type == 2:
        best_window_size = 50 * 60
        agent = SlidingWindowUCBAgent(window_size=best_window_size * 60)
    elif agent_type == 3:
        agent = DiscountedUCBAgent(gamma=0.999)
    agent.initialize(env.n_camera)

    env.reset(mode='train')

    inference_times = []

    for t in tqdm(range(env.length), initial=2):
        start_time = time.time()
        action = agent.get_action()
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        reward, state, stop = env.step(action)
        agent.update(action, state)

        if stop:
            break

    total_reward = env.get_total_reward()
    print(f'=== TRAINING===')
    print('[total reward]:', total_reward)

    return np.mean(inference_times)


if __name__ == '__main__':
    print("Enter the agent you want to test: ")
    inp = int(input('1. UCB-1 \n2. SW-UCB \n3. D-UCB\n4. Time experiment\n'))
    if inp == 1:
        SWUCBOpt(1)
    elif inp == 2:
        inp2 = int(input('Find optimal window size? (1. Yes, 2. No)'))
        if inp2 == 1:
            SWUCBExperiment()
        elif inp2 == 2:
            SWUCBOpt(2)
    elif inp == 3:
        inp2 = int(input('Find optimal gamma? (1. Yes, 2. No)'))
        if inp2 == 1:
            DiscountedUCBExperiment()
        elif inp2 == 2:
            SWUCBOpt(3)
    elif inp == 4:
        sw_ucb_inference_times = timeexperiment(2)
        ucb1_inference_times = timeexperiment(1)
        d_ucb_inference_times = timeexperiment(3)

        # Plot the average inference times for each agent as a bar plot
        agents = ['SW-UCB', 'UCB-1', 'D-UCB']
        avg_inference_times = [sw_ucb_inference_times,
                               ucb1_inference_times,
                               d_ucb_inference_times]

        print(avg_inference_times)

        # Black and white color palette
        colors = ['#333333', '#666666', '#999999']

        plt.bar(agents, avg_inference_times, color=colors, edgecolor='black',
                linewidth=1)
        plt.xlabel('Agent', fontsize=12)
        plt.ylabel('Average Inference Time (s)', fontsize=10)
        plt.title(
            'Average Inference Time for SW-UCB, UCB-1, and D-UCB during Training',
            fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('UCB_AverageInferenceTime_Training_BW.png')
        plt.show()

"""
Baseline:
====== TESTING ======
[total reward]: 0.506


Run 1 SW-UCB best window size = 50 * 60:
== TRAINING===
[total reward]: 0.563                      
====== TESTING======
[total reward]: 0.524

Difference Strong Baseline = Run 1 - Baseline = 0.524 - 0.506 = 0.018
Percentage growth = (Difference / Baseline) x 100 = 0.018 / 0.506 x 100 = 3.6%

Difference Weak Baseline = Run 1 - Baseline = 0.524 - 0.317 = 0.207
Percentage growth = (Difference / Baseline) x 100 = 0.207 / 0.317 x 100 = 65.3%
Growth: 65.3%.


Run 2 UCB1:                                             
====== TESTING======
[total reward]: 0.499

Difference Strong Baseline = Run 2 - Baseline = 0.499 - 0.506 = -0.007
Percentage growth = (Difference / Baseline) x 100 = -0.007 / 0.506 x 100 = -1.4%

Difference Weak Baseline = Run 2 - Baseline = 0.499 - 0.317 = 0.182
Percentage growth = (Difference / Baseline) x 100 = 0.182 / 0.317 x 100 = 57.4%

Run 3 D-UCB:
====== TESTING======
[total reward]: 0.559
Best gamma: 0.999

Difference Strong Baseline = Run 3 - Baseline = 0.559 - 0.506 = 0.053
Percentage growth = (Difference / Baseline) x 100 = 0.053 / 0.506 x 100 = 10.5%

Difference Weak Baseline = Run 3 - Baseline = 0.559 - 0.317 = 0.242
Percentage growth = (Difference / Baseline) x 100 = 0.242 / 0.317 x 100 = 76.2%
"""

"""
Time experiment: Average inference time for each agent during training
[6.7903322202188e-06, 0.00015759250080143964, 1.392593538319623e-05]
UCB-1 SW-UCB D-UCB
"""



