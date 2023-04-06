# agent script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023

import numpy as np
import torch

class baselineAgent:

    def __init__(self, record_length=10):
        self.records = None
        self.record_length = record_length
        self.moving_avg = 0
        self.model = None
        print("in baselineAgent __init__")


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
        
        
        
class LSTMAgent:
    def __init__(self, model, input_size, record_length=10):
        self.model = model
        self.records = None
        self.record_length = record_length
        self.input_size = input_size

    def preprocess_data(self, records, input_size):
        # Convert the records into a NumPy array
        data = np.array(records)
        print("Data shape:", data.shape)
        print("input size:", input_size)
        # Normalize or scale the data if necessary
        data_normalized = data / np.max(np.abs(data), axis=0)

        # Reshape the data to match the input shape expected by the LSTM model
        seq_length = data_normalized.shape[0]
        input_data = np.zeros((seq_length, 1, input_size))
        input_data[:, 0, :] = data_normalized[:, :]

        return input_data

    def get_action(self, state):
        self.add_record(state)

        input_data = self.preprocess_data(self.records, self.input_size)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        print("Input tensor shape:", input_tensor.shape)

        output = self.model(input_tensor)

        # Get the index of the maximum score as the action
        action_index = np.argmax(output.detach().numpy())
        return action_index

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
