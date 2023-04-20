# agent script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023


from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
import numpy as np

class baselineAgent:

    def __init__(self, theta=0, agent_type='strong'):
        assert agent_type in ['simple', 'strong']
        self.agent_type = agent_type
        self.prev_state = None
        self.theta = theta

    def get_action(self, state):
        # store previous state
        self.prev_state = state

        if self.agent_type=='simple':
            # random action
            action = np.random.rand(len(state))

        elif self.agent_type=='strong':
            # use previous states as scores (-1 is replaced by the learned param theta)
            action = np.array([v if v>=0 else self.theta for v in self.prev_state])

        return action

class LSTM_RNN_Agent:
    def __init__(self, record_length=10, input_dim=1, hidden_dim=32,
                 learning_rate=0.001, dropout_rate=0.0,
                 l1_reg=0.0, l2_reg=0.0, epsilon=0.1):
        self.records = [[] for _ in range(input_dim)]
        self.record_length = record_length
        self.moving_avg = 0
        self.epsilon = epsilon
        self.input_dim = input_dim
        self.model = self.build_model(input_dim, hidden_dim, learning_rate,
                                      dropout_rate, l1_reg, l2_reg)

    def build_model(self, input_dim, hidden_dim, learning_rate,
                    dropout_rate, l1_reg, l2_reg):
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(LSTM(hidden_dim,
                           input_shape=(self.record_length, input_dim)))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

            model.add(Dense(input_dim, activation='softmax',
                            kernel_regularizer=L1L2(l1=l1_reg, l2=l2_reg)))
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(learning_rate=learning_rate))
            return model

    def get_action(self, state):
        self.add_record(state)
        imputed_state = self.impute_missing_values(state)

        # Make sure records have enough data to form the required input shape
        if len(self.records[0]) < self.record_length:
            action = np.random.rand(len(imputed_state))
        else:
            input_data = np.array(self.records).T.reshape(1,
                                                          self.record_length,
                                                          self.input_dim)

            # Exploration vs. Exploitation
            if np.random.rand() < self.epsilon:
                action = np.random.rand(len(imputed_state))
            else:
                action = self.model.predict(input_data)

            action = action.flatten() / action.sum()
        return action

    def add_record(self, state):
        if not self.records or not self.records[0]:  # Check for empty list of lists
            self.records = [[] for _ in range(len(state))]
        for r, s in zip(self.records, state):
            r.append(s)
            if len(r) > self.record_length:
                r.pop(0)

    def clear_records(self):
        self.records = [[] for _ in range(len(self.records))]  # Clear records while maintaining the structure

    def train_on_batch(self, X, y):
        with tf.device('/gpu:0'):
            loss, accuracy = self.model.train_on_batch(X, y), None
            metrics_values = self.model.evaluate(X, y, verbose=0)
            if isinstance(metrics_values, float):
                # If metrics_values is a float, set accuracy directly
                accuracy = metrics_values
            else:
                # Otherwise, calculate accuracy from metrics_values
                accuracy = metrics_values[1]
            return loss, accuracy

    def impute_missing_values(self, state):
        """Impute missing values in the state with the mean of historical records."""
        imputed_state = np.copy(state)
        for i, s in enumerate(state):
            if s == -1:
                imputed_state[i] = np.mean(self.records[i])
        return imputed_state


class SlidingWindowUCBAgent:
    def __init__(self, c=5, window_size=100):
        self.counts = None
        self.values = None
        self.c = c
        self.window_size = window_size
        self.recent_rewards = None
        self.recent_counts = None
        self.total_time_steps = 0

    def initialize(self, n_actions):
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)
        self.recent_rewards = [deque(maxlen=self.window_size) for _ in range(n_actions)]
        self.recent_counts = [deque(maxlen=self.window_size) for _ in range(n_actions)]

    def get_action(self):
        if self.counts.min == 0:
        # action = np.random. choice(np.where(self.counts == 0) [0])
            idx = np.random.choice(np.where(self.counts == 0)[0])
            action = np.zeros(len(self.values))
            action[idx] = 1
        else:
            min_time_steps = min(self.total_time_steps, self.window_size)
            recent_counts_sum = np.array([sum(counts) for counts in self.recent_counts])
            ucb_values = self.values + self.c * np.sqrt(2 * np.log(min_time_steps) / recent_counts_sum)
            action = ucb_values
        return action


    def update(self, actions, state):
        self.total_time_steps += 1
        for i, reward in enumerate(state):
            if reward >= 0:
                self.counts[i] += 1
                self.recent_rewards[i].append(reward)
                self.recent_counts[i].append(1)

                # Calculate the average reward based on the sliding window
                avg_reward = sum(self.recent_rewards[i]) / sum(self.recent_counts[i])
                self.values[i] = avg_reward
            else:
                self.counts[i] += 0
                # self.recent_counts[i].append(0)

class DiscountedUCBAgent:
    def __init__(self, c=5, gamma=0.99):
        self.counts = None
        self.values = None
        self.c = c
        self.gamma = gamma
        self.total_time_steps = 0

    def initialize(self, n_actions):
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)

    def get_action(self):
        if self.counts.min == 0:
        # action = np.random. choice(np.where(self.counts == 0) [0])
            idx = np.random.choice(np.where(self.counts == 0)[0])
            action = np.zeros(len(self.values))
            action[idx] = 1
        else:
            ucb_values = self.values + self.c * np.sqrt(2 * np.log(self.total_time_steps) / self.counts)
            action = ucb_values
        return action

    def update(self, actions, state):
        self.total_time_steps += 1
        for i, reward in enumerate(state):
            if reward >= 0:
                self.counts[i] += 1
                alpha = 1 / self.counts[i]
                self.values[i] = (1 - alpha) * self.values[i] + alpha * reward * pow(self.gamma, self.total_time_steps - self.counts[i])
            else:
                self.counts[i] += 0


class UCBAgent:
    def __init__(self, c=5):
        self.counts = None
        self.values = None
        self.c = c
        self.total_time_steps = 0

    def initialize(self, n_actions):
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)

    def get_action(self):
        if self.counts.min == 0:
        # action = np.random. choice(np.where(self.counts == 0) [0])
            idx = np.random.choice(np.where(self.counts == 0)[0])
            action = np.zeros(len(self.values))
            action[idx] = 1
        else:
            ucb_values = self.values + self.c * np.sqrt(2 * np.log(self.total_time_steps) / self.counts)
            action = ucb_values
        return action

    def update(self, actions, state):
        self.total_time_steps += 1
        for i, reward in enumerate(state):
            if reward >= 0:
                self.counts[i] += 1
                self.values[i] = self.values[i] + (1 / self.counts[i]) * (reward - self.values[i])
            else:
                self.counts[i] += 0
