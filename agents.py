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
from tensorflow.keras.layers import Bidirectional
import numpy as np


class LSTM_RNN_Agent:
    def __init__(self, record_length=10, input_dim=1, hidden_dim=32,
                 learning_rate=0.001, n_lstm_layers=1, dropout_rate=0.0,
                 l1_reg=0.0, l2_reg=0.0, epsilon=0.1):
        self.records = [[] for _ in range(input_dim)]
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




class Bidirectional_LSTM_RNN_Agent:
    def __init__(self, record_length=10, input_dim=1, hidden_dim=32,
                 learning_rate=0.001, dropout_rate=0.0,
                 l1_reg=0.0, l2_reg=0.0, epsilon=0.1):
        self.records = [[] for _ in range(input_dim)]
        self.record_length = record_length
        self.moving_avg = 0
        self.epsilon = epsilon
        self.model = self.build_model(input_dim, hidden_dim, learning_rate,
                                      dropout_rate, l1_reg, l2_reg)

    def build_model(self, input_dim, hidden_dim, learning_rate,
                    dropout_rate, l1_reg, l2_reg):
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Bidirectional(LSTM(hidden_dim,
                                         input_shape=(self.record_length, input_dim))))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

            model.add(Dense(input_dim, activation='softmax',
                            kernel_regularizer=L1L2(l1=l1_reg, l2=l2_reg)))
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(learning_rate=learning_rate))
            return model

    # The rest of the class remains unchanged


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
            ucb_values = np.zeros_like(self.values)
            ucb_values[action] = 1
        else:
            ucb_values = self.values + self.c * np.sqrt(2 * np.log(self.counts.sum()) / self.counts)
        return ucb_values

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

