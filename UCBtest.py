"""
RNNtest script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
author: Jasper Bruin @ UvA-MNS
date: 23/02/2023
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import SlidingWindowUCBAgent, UCBAgent, DiscountedUCBAgent, baselineAgent
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
        inference_time = (end_time - start_time) * 1000
        inference_times.append(inference_time)

        reward, state, stop = env.step(action)
        agent.update(action, state)

        if stop:
            break

    total_reward = env.get_total_reward()
    print(f'=== TRAINING===')
    print('[total reward]:', total_reward)

    return np.mean(inference_times)

def robustness_test(agent_type, budget_ratios):
    if agent_type == 1:
        print("UCB-1")
    elif agent_type == 2:
        print("SW-UCB")
    elif agent_type == 3:
        print("D-UCB")
    elif agent_type == 4:
        print("Simple Baseline")
    elif agent_type == 5:
        print("Strong Baseline")

    np.random.seed(0)

    best_window_size = 50 * 60  # Best window size obtained from previous experiments
    best_gamma = 0.999  # Best gamma obtained from previous experiments

    rewards = []

    for budget_ratio in budget_ratios:
        env = ROFARS_v1(budget_ratio=budget_ratio)
        env.reset(mode='test')

        if agent_type == 1:
            agent = UCBAgent()
            agent.initialize(env.n_camera)


            for t in range(env.length):
                action = agent.get_action()
                reward, state, stop = env.step(action)
                agent.update(action, state)

                if stop:
                    break

            total_reward = env.get_total_reward()
            rewards.append(total_reward)

        elif agent_type == 2:
            agent = SlidingWindowUCBAgent(window_size=best_window_size)
            agent.initialize(env.n_camera)


            for t in range(env.length):
                action = agent.get_action()
                reward, state, stop = env.step(action)
                agent.update(action, state)

                if stop:
                    break

            total_reward = env.get_total_reward()
            rewards.append(total_reward)

        elif agent_type == 3:
            agent = DiscountedUCBAgent(gamma=best_gamma)
            agent.initialize(env.n_camera)


            for t in range(env.length):
                action = agent.get_action()
                reward, state, stop = env.step(action)
                agent.update(action, state)

                if stop:
                    break

            total_reward = env.get_total_reward()
            rewards.append(total_reward)


        elif agent_type == 4:

            agent = baselineAgent(agent_type='simple', theta=0)
            # give random scores as the initial action
            init_action = np.random.rand(env.n_camera)
            reward, state, stop = env.step(init_action)

            for t in tqdm(range(env.length), initial=2):

                action = agent.get_action(state)
                reward, state, stop = env.step(action)


                # do sth to update your algorithm here

                if stop:
                    break
            total_reward = env.get_total_reward()
            rewards.append(total_reward)
        elif agent_type == 5:
            agent = baselineAgent(agent_type='strong', theta=0)
            # give random scores as the initial action
            init_action = np.random.rand(env.n_camera)
            reward, state, stop = env.step(init_action)

            for t in tqdm(range(env.length), initial=2):

                action = agent.get_action(state)
                reward, state, stop = env.step(action)
                # do sth to update your algorithm here

                if stop:
                    break

            total_reward = env.get_total_reward()
            rewards.append(total_reward)


    return budget_ratios, rewards

