import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import UCBAgent

# TODO: Exponential recency weighted average or sliding window
# TODO: Not sure about this is how to do the sliding window implementation
def sliding_window(data, window_size):
    result = []
    for i in range(len(data) - window_size + 1):
        result.append(data[i:i + window_size])
    return result

np.random.seed(0)
env = ROFARS_v1()
agent = UCBAgent(c=3)
n_episode = 30

# Initialize the UCB Agent
agent.initialize(env.n_camera)

# training
for episode in range(n_episode):
    env.reset(mode='train')

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action()
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(action, state)

        if stop:
            break

    print(f'=== TRAINING episode {episode} ===')
    print('[total reward]:', env.get_total_reward())

# testing
env.reset(mode='test')

for t in tqdm(range(env.length), initial=2):
    action = agent.get_action()
    reward, state, stop = env.step(action)

    if stop:
        break

print(f'====== TESTING ======')
print('[total reward]:', env.get_total_reward())