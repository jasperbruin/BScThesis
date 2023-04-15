# main script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023

"""
-   Different learning rates
-   Batch size
-   Number of hidden layers
-   Number of LSTM layers: in build_model
-   Regularization: dropout and L1L2
-   Use a validation set: Currently, the code only trains and tests the model.
    To avoid overfitting while optimizing hyperparameters, consider splitting the
    training data into a training set and a validation set. Then, during the
    hyperparameter search, evaluate the model on the validation set.
"""


import numpy as np
from tqdm import tqdm
from agents import Bidirectional_LSTM_RNN_Agent
from rofarsEnv import ROFARS_v1

np.random.seed(0)

env = ROFARS_v1(length=36*10)
agent = Bidirectional_LSTM_RNN_Agent(
    hidden_dim=32,
    learning_rate=0.01,
    dropout_rate=0.5,
    l1_reg=0.001,
    l2_reg=0.001
)

n_episode = 30
batch_size = 32
X_batch = []
y_batch = []

# training
for episode in range(n_episode):
    # Reset loss and accuracy lists
    training_loss = []
    training_accuracy = []

    env.reset(mode='train')
    agent.clear_records()
    init_action = np.random.rand(env.n_camera)
    reward, state, stop = env.step(init_action)

    for t in tqdm(range(env.length), initial=2):
        action = agent.get_action(state)  # Corrected: Pass the state as input
        reward, state, stop = env.step(action)

        if len(agent.records[0]) == agent.record_length:
            X = np.array(agent.records).T.reshape(-1, agent.record_length, 1)
            y = np.eye(env.n_camera)[np.argmax(action, axis=0)]

            X_batch.append(X)
            y_batch.append(y)

            if len(X_batch) == batch_size:
                X_batch = np.concatenate(X_batch, axis=0)
                y_batch = np.concatenate(y_batch, axis=0)
                loss, accuracy = agent.train_on_batch(X_batch, y_batch)

                # Append loss and accuracy inside the if statement
                training_loss.append(loss)
                training_accuracy.append(accuracy)

                X_batch = []
                y_batch = []

        if stop:
            break

    print(f'=== TRAINING episode {episode} ===')
    print('[total reward]:', env.get_total_reward())
    print('[average loss]:', np.mean(training_loss))
    print('[average accuracy]:', np.mean(training_accuracy))

# testing
env.reset(mode='test')
agent.clear_records()
init_action = np.random.rand(env.n_camera)
reward, state, stop = env.step(init_action)

for t in tqdm(range(env.length), initial=2):

    action = agent.get_action(state)
    reward, state, stop = env.step(action)

    if stop:
        break

print(f'====== TESTING ======')
print('[total reward]:', env.get_total_reward())