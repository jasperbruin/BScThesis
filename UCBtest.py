import numpy as np
from tqdm import tqdm
from rofarsEnv import ROFARS_v1
from agents import UCBAgent

np.random.seed(0)

env = ROFARS_v1()
agent = UCBAgent(c=2)
n_episode = 30

# Initialize the UCB Agent
agent.initialize(env.n_camera)

# training
for episode in range(n_episode):

    env.reset(mode='train')

    for t in tqdm(range(env.length), initial=2):

        action = np.zeros(env.n_camera)
        action[agent.get_action()] = 1
        reward, state, stop = env.step(action)

        # Update the UCB Agent
        agent.update(agent.get_action(), reward)

        if stop:
            break

    print(f'=== TRAINING episode {episode} ===')
    print('[total reward]:', env.get_total_reward())

# testing
env.reset(mode='test')

for t in tqdm(range(env.length), initial=2):

    action = np.zeros(env.n_camera)
    action[agent.get_action()] = 1
    reward, state, stop = env.step(action)

    if stop:
        break

print(f'====== TESTING ======')
print('[total reward]:', env.get_total_reward())