# environment for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# train.txt and test.txt are required
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023

import pandas as pd
import numpy as np

class ROFARS_v1:
    def __init__(self, length=3600*12, n_camera=10, budget_ratio=0.5, data_path='data/train_test.txt'):
        # 43,200 seconds = 12 hours = 1/2 day
        self.length = length
        # 10 cameras by default
        self.n_camera = n_camera
        # set ratio
        self.budget_ratio = budget_ratio
        # generate cameras' stream
        self.train_cameras, self.test_cameras = self.init_cameras(data_path)
        self.reset()

    def reset(self, mode='train'):
        # train/test mode
        if mode == 'train':
            self.cameras = self.train_cameras
        elif mode == 'test':
            self.cameras = self.test_cameras
        self.index = 0
        self.rewards = []

    def init_cameras(self, path):
        data = pd.read_csv(path, sep=' ', names=['count', 'date', 'time'])
        assert self.length * self.n_camera * 2 <= len(data)
        train_cameras, test_cameras = [], []
        # assign the first half to training data
        train_index = range(self.n_camera)
        # assign the second half to testing data
        test_index = range(self.n_camera, self.n_camera * 2)
        for i in train_index:
            train_cameras.append(data[self.length*i:self.length*(i+1)].reset_index(drop=True))
        for i in test_index:
            test_cameras.append(data[self.length*i:self.length*(i+1)].reset_index(drop=True))
        return train_cameras, test_cameras

    def get_total_reward(self):
        return np.mean(self.rewards).round(3)

    def step(self, action):
        # adding small noise to the action for randomness when all values are the same
        action += np.random.rand(*action.shape)/100
        # action is a vector of scores with n_camera-dimension
        # -1 refers to the unchecked camera state
        state = np.ones(self.n_camera)*(-1)
        # get the index of cameras for check according to the budget
        check_index = np.argsort(action)[::-1][:int(self.budget_ratio * self.n_camera)]
        for i in check_index:
            state[i] = self.cameras[i]['count'][self.index]

        # collect number of total faces as the reward
        reward = state[state != -1].sum()/self.n_camera
        self.rewards.append(reward)

        # check if stop
        if self.index == self.length-1:
            stop = True
        else:
            stop = False

        # step
        self.index += 1

        return reward, state, stop

if __name__ == "__main__":
    env = ROFARS_v1()
