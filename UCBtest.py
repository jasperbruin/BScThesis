import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import UCBAgent

def sliding_window(data, window_size):
    result = []
    for i in range(len(data) - window_size + 1):
        result.append(data[i:i + window_size])
    return result

np.random.seed(0)

env = ROFARS_v1()
agent = UCBAgent(c=2)
n_episode = 30
window_size_range = range(1, 101)

# Initialize the UCB Agent
agent.initialize(env.n_camera)

# training
for episode in range(n_episode):
    for window_size in window_size_range:
        env.reset(mode='train')

        for t in tqdm(range(env.length), initial=2):
            state = env.get_state(window_size)
            traces = sliding_window(state, window_size)
            traces_with_random_scores = [[trace + np.random.rand() for trace in trace_array] for trace_array in traces]


            action = np.zeros(env.n_camera)
            action[agent.get_action()] = 1
            reward, state, stop = env.step(action)

            # Update the UCB Agent
            agent.update(agent.get_action(), reward)

            if stop:
                break

        print(f'=== TRAINING episode {episode}, window size {window_size} ===')
        print('[total reward]:', env.get_total_reward())

# testing
for window_size in window_size_range:
    env.reset(mode='test')

    for t in tqdm(range(env.length), initial=2):
        state = env.get_state(window_size)
        traces = sliding_window(state, window_size)
        traces_with_random_scores = [trace + np.random.rand() for trace in traces]

        action = np.zeros(env.n_camera)
        action[agent.get_action()] = 1
        reward, state, stop = env.step(action)

        if stop:
            break

    print(f'====== TESTING, window size {window_size} ======')
    print('[total reward]:', env.get_total_reward())
