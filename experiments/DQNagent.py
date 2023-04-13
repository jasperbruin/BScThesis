import numpy as np
from tqdm import tqdm
from agents import DQNAgent
from rofarsEnv import ROFARS_v1

np.random.seed(0)
env = ROFARS_v1()
state_size = env.n_camera
action_size = env.n_camera

learning_rate = 0.005
gamma = 0.90
epsilon_decay = 0.99
epsilon_min = 0.02

agent = DQNAgent(state_size, action_size, learning_rate=learning_rate, gamma=gamma, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

n_episode = 30
window_size = 36 * 10

# training
for episode in range(n_episode):

    env.reset(mode='train')
    state = np.array(env.get_state(window_size)).reshape(1, -1)  # Modified line
    reward, next_state, stop = env.step(agent.get_action(state))

    for t in tqdm(range(env.length), initial=2):

        action = agent.get_action(state)
        reward, next_state, stop = env.step(action)

        if stop:
            break

    print(f'=== TRAINING episode {episode} ===')
    print('[total reward]:', env.get_total_reward())

# testing
env.reset(mode='test')
state = np.array(env.get_state(window_size)).reshape(1, -1)  # Modified line
reward, next_state, stop = env.step(agent.get_action(state))

for t in tqdm(range(env.length), initial=2):

    action = agent.get_action(state)
    reward, next_state, stop = env.step(action)

    if stop:
        break

print(f'====== TESTING ======')
print('[total reward]:', env.get_total_reward())
