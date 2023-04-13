# agent script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023
import random
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Masking
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
import numpy as np


class baselineAgent:

    def __init__(self, record_length=10):
        self.records = None
        self.record_length = record_length
        self.moving_avg = 0
        self.model = None


    def get_action(self, state):
        # add t orecord
        self.add_record(state)

        # random action
        #action = np.random.rand(len(state))

        # sum of historical records as scores (-1 is taken as 1 to encourage exploration)
        action = np.array([np.sum(np.abs(r)) for r in self.records])

        # normalization as the final step
        action = action/action.sum()
        return action

    def add_record(self, state):
        # init records w.r.t. the size of states
        if self.records is None:
            self.records = [[] for _ in range(len(state))]
        # insert records
        for r, s in zip(self.records, state):
            r.append(s)
            # discard oldest records
            if len(r) > self.record_length:
                r.pop(0)

    def clear_records(self):
        self.records = None


class LSTM_RNN_Agent:
    def __init__(self, record_length=10, input_dim=1, hidden_dim=32,
                 learning_rate=0.001, n_lstm_layers=1, dropout_rate=0.0,
                 l1_reg=0.0, l2_reg=0.0, epsilon=0.1):
        self.records = [[] for _ in range(input_dim)]  # Initialize as an empty list of lists
        self.record_length = record_length
        self.moving_avg = 0
        self.epsilon = epsilon
        self.model = self.build_model(input_dim, hidden_dim, learning_rate,
                                      n_lstm_layers, dropout_rate, l1_reg,
                                      l2_reg)

    def build_model(self, input_dim, hidden_dim, learning_rate, n_lstm_layers,
                    dropout_rate, l1_reg, l2_reg):
        with tf.device('/gpu:0'):
            model = Sequential()
            for i in range(n_lstm_layers):
                if i == 0:
                    model.add(LSTM(hidden_dim,
                                   input_shape=(self.record_length, input_dim),
                                   return_sequences=True if n_lstm_layers > 1 else False))
                else:
                    model.add(LSTM(hidden_dim,
                                   return_sequences=True if i < n_lstm_layers - 1 else False))
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
        input_data = np.array(self.records).T.reshape(-1, self.record_length, 1)

        # Exploration vs. Exploitation
        if np.random.rand() < self.epsilon:
            action = np.random.rand(len(state))
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

class UCBAgent:
    def __init__(self, c=2):
        self.counts = None
        self.values = None
        self.c = c

    def initialize(self, n_actions):
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)

    def get_action(self):
        if self.counts.min() == 0:
            action = np.random.choice(np.where(self.counts == 0)[0])
        else:
            ucb_values = self.values + self.c * np.sqrt(2 * np.log(self.counts.sum()) / self.counts)
            action = np.argmax(ucb_values)
        return action

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

class SimpleRNNAgent:
    def __init__(self, record_length=10):
        self.records = None
        self.record_length = record_length
        self.moving_avg = 0
        self.epsilon = 0.1
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        # Removed Masking layer
        self.model.add(SimpleRNN(units=4, activation='tanh', input_shape=(None, 1)))  # Reduced hidden units
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.001))

    def get_action(self, state):
        # Add to record
        self.add_record(state)

        # Impute missing values
        imputed_state = self.impute_missing_values(state)

        # Exploration vs Exploitation
        if np.random.rand() < self.epsilon:
            action = np.random.rand(len(state))
        else:
            action = self.model.predict(np.expand_dims(imputed_state, axis=-1))

        # normalization as the final step
        action = action / action.sum()
        return action.squeeze()

    def add_record(self, state):
        # init records w.r.t. the size of states
        if self.records is None:
            self.records = [[] for _ in range(len(state))]
        # insert records
        for r, s in zip(self.records, state):
            r.append(s)
            # discard oldest records
            if len(r) > self.record_length:
                r.pop(0)

    def clear_records(self):
        self.records = None

    def impute_missing_values(self, state):
        """ Impute missing values in the state with the mean of historical records.
            I want to experiment with different missing input strategies."""
        imputed_state = np.copy(state)
        for i, s in enumerate(state):
            if s == -1:
                imputed_state[i] = np.mean(self.records[i])
        return imputed_state


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size)
        else:
            act_values = self.model.predict(state)
            return act_values[0]