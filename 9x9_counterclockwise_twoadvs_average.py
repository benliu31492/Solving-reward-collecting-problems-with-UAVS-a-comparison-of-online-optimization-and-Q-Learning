#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:31:14 2020

@author: madeleine
"""

from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Dense, Activation, Dropout, Input, Lambda, Add, Subtract
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Lambda, Add, Subtract
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from gym.wrappers import Monitor
import collections

import time
import math
import matplotlib
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import shutil
import pickle

import random
from random import seed
from random import randint

## Setting seed
seed(213)
# Gray scale marks for cells
advs_mark = 0.25
agent_visited_mark = 0.75
advs_visited_mark = 0.85
flag_mark = 0.65
agent_mark = 0.55

# Actions dictionary
actions_dict = {
    0: 'left',
    1: 'up',
    2: 'right',
    3: 'down',
}
# number of actions, in this case there is 4
num_actions = len(actions_dict)

# maze is a 2d Numpy array of floats valued at 1.0
# 1.0 corresponds to a free cell that our UAV can traverse through
# establishes the maze object for our reinforcement learning
class Tmaze(object):
    """
    Tour De Flags maze object
    maze: a 2d Numpy array of 0's and 1's
        1.00 - a free cell
        0.65 - flag cell
        0.50 - agent cell
    agent: (row, col) initial agent position (defaults to (0,0))
    flags: list of cells occupied by flags
    """
    def __init__(self, maze, flags, advs, agent=(0,0), target=None):
        # establishes the initial positions for our adversaries
        self._advs = list(advs)
        # estalishes the initial position for our adversary
        self._agent = agent
        # establishes our 2d Numpy array maze
        self._maze = np.array(maze)
        # establishes our action space as 4 discrete possibilities
        self.action_space = gym.spaces.Discrete(4)
        # initializes our flags' positions
        self._flags = list(flags)
        # set to keep track of uncaptured flags
        self.uncaptured_flags = set(flags)
        # initializes our adversaries as a set
        self.advs = list(advs)
        # establishes nrows and ncols as the # of rows and columns in our maze
        nrows, ncols = self._maze.shape
        # If we don't establish a target, creates a default one
        if target is None:
            # default target cell where the agent to deliver the "flags"
            # default is bottom right corner cell
            self.target = (nrows-1, ncols-1)
        # creates dictionary to keep track of agent path with highest reward
        # in each run
        self.best_agent_visited = {}
        # initializes highest reward as 0
        self.best_total_reward = 0
        # initializes which win had highest reward as 0
        self.best_win_count = 0
        # creates list to keep track of agent actions with highest reward each run
        self.best_actions = []
        # resets our maze with initial agent and adversary positions
        self.reset(self._agent,self._advs)
    # function to reset our maze
    def reset(self, agent, advs):
        # initializes agent position
        self.agent = agent
        # initializes adversary position
        self.advs = advs
        # creates a copy of our maze
        self.maze = np.copy(self._maze)
        # initializes our flags
        self.flags = list(self._flags)
        # set to keep track of uncaptured flags
        self.uncaptured_flags = set(self._flags)
        # extracts the number of rows and columns in our maze
        nrows, ncols = self.maze.shape
        # extracts the row and column where our agent is located
        row, col = agent
        # marks the maze cell where our agent is located
        self.maze[row, col] = agent_mark
        # initializes our starting state after reset
        self.state = ((row, col), 'start')
        # initializes the base, which in a 5x5 grid maze is 5
        self.base = np.sqrt(self.maze.size)
        # initializes agent visited cells as 0 for all free cells
        self.agent_visited = dict(((r,c),0) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0)
        # keeps track of where our agent starts
        self.agent_visited[self.agent] += 1
        # initializes total reward
        self.total_reward = 0
        # creates list to keep track of agent actions
        self.agent_actions = []
        # initializes minimum reward before game is stopped and reset
        self.min_reward = -100
        # establishes rewards for actions in a game
        self.reward = {
            # being captured by the adversary results in -1000 points, which
            # will end the game
            'captured': -1000,
            # capturing the flag results in 200 points
            'flag':     200,
            # each step results in -1 points
            'valid':   -1,
        }
    # defines the process of each step/action for the agent
    def act(self, action):
        # updates the state after each step/action
        self.update_state(action)
        # updates the reward for each step/action
        reward = self.get_reward()
        # updates the total reward after each step/action
        self.total_reward += reward
        # updates the status of the game after each step/action
        status = self.game_status()
        # observes the state of our environment/maze after each step/action
        env_state = self.observe()
        # returns the environment state, reward, and status changes after
        # each step/action
        return env_state, reward, status
    # definies the process of getting a reward
    def get_reward(self):
        # initializes the agent and mode from the state
        agent, mode = self.state
        # if the agent is in the target cell, return 100 points
        if agent == self.target:
            if len(self.uncaptured_flags)==0:
                return 100
        #   else:
        #        return -1000
        # if the agent is in the same cell as the adversary, receive
        # -1000 points
        if agent in self.advs:
            return self.reward['captured']
        # if the agent captures the flag, receive 200 points
        elif agent in self.uncaptured_flags:
            return self.reward['flag']
        # if the agent takes a step without ending the game or capturing the
        # flag, receive -1 points
        elif mode == 'valid':
            return self.reward['valid']
        # return 0 points when the game is reset
        else:
            return 0
    # definie updating the state of the game after each action
    def update_state(self, action):
        # extract the # of rows and columns of our maze
        nrows, ncols = self.maze.shape
        # extracts the agent position and mode of our game from the game state
        (nrow, ncol), nmode = agent, mode = self.state
        # if an agent is in the same cell as a flag, remove the flag from our
        # updates set of uncaptured flags
        if agent in self.uncaptured_flags:
            self.uncaptured_flags.remove(agent)
        # retrieves valid agent actions
        valid_actions = self.agent_valid_actions(agent)
        # if there are no valid actions, return the mode as blocked
        if action not in valid_actions:
            action = random.choice(valid_actions)
            if action == 0:    # move left
                ncol -= 1
            elif action == 1:  # move up
                nrow -= 1
            elif action == 2:    # move right
                ncol += 1
            elif action == 3:  # move down
                nrow += 1
        # updates location of the agent based on valid actions
        elif action in valid_actions:
            nmode = 'valid'
            if action == 0:    # move left
                ncol -= 1
            elif action == 1:  # move up
                nrow -= 1
            elif action == 2:    # move right
                ncol += 1
            elif action == 3:  # move down
                nrow += 1
                # keeps track of action
        self.agent_actions.append(action)
        # new location for agent
        agent = (nrow, ncol)
        # keeps track of where our agent has visited
        self.agent_visited[agent] += 1
        for c in range(len(self.advs)):

            # retrieve the valid adversary action based off of its current location
            advs_action = self.advs_valid_actions(cell = self.advs[c],flag = self.flags[c])
            # extract the location of the adversary
            advs_row, advs_col = self.advs[c]
            # update the location of the adversary based on chosen action
            if advs_action == 0:    # move left
                advs_col -= 1
            elif advs_action == 1:  # move up
                advs_row -= 1
            elif advs_action == 2:    # move right
                advs_col += 1
            elif advs_action == 3:  # move down
                advs_row += 1
            # establishes new location of adversary
            self.advs[c] = (advs_row,advs_col)

        # updates the state of the game
        self.state = (agent, nmode)
    # defines how the status of the game is updated
    def game_status(self):
        # if we have exceeded the minimum reward of the game,
        # update the state to show we have lost the game
        if self.total_reward < self.min_reward:
            return 'lose'
        # update the agent and mode from the game state
        agent, mode = self.state
        # if the agent has reached the target cell and captured all flags,
        # update the state as having won the game
        if agent == self.target and len(self.uncaptured_flags) == 0:
            return 'win'
            # if we reach the target cell without having captured all flags,
            # we have lost the game
            #else:
             #   return 'lose'
        # if we have not reached the target cell, we are still playing
        else:
            return 'ongoing'
    # defines how the game is observed
    def observe(self):
        # takes a canvas shot of the maze
        canvas = self.draw_env()
        # turns the canvas into a 1x25 array
        env_state = canvas.reshape((1, -1))
        return env_state
    # defines how we draw the environment of the game
    def draw_env(self):
        # makes a copy of our maze
        canvas = np.copy(self.maze)
        # extracts the # of rows and columns of our maze
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the flags
        for r,c in self.flags:
            canvas[r,c] = flag_mark
        # draw the agent
        agent, mode = self.state
        canvas[agent] = agent_mark
        return canvas
    # defines how we calculate valid adversary actions
    def advs_valid_actions(self, cell=None, flag=None):
        # retrievevs location of our adversary
        advs_row, advs_col = cell
        # rerieves location of our flag
        flag_row, flag_col = flag
        # code below ensures our adversary is always moving clockwise
        # around the flag
        if advs_col < flag_col and advs_row <= flag_row:
            actions = 3
        if advs_row < flag_row and advs_col >= flag_col:
            actions = 0
        if advs_col > flag_col and advs_row >= flag_row:
            actions = 1
        if advs_row > flag_row and advs_col <= flag_col:
            actions = 2
        return actions
    # defines how valid agent actions are calculated
    def agent_valid_actions(self, cell=None):
        # if we do not supply where agent is located, calculate based on
        # location of agent from the game state
        if cell is None:
            (row, col), mode = self.state
        else:
            row, col = cell
        # possible actions
        actions = [0, 1, 2, 3]
        # retrieves the # of rows and coluns of our maze
        nrows, ncols = self.maze.shape
        # code below ensures our agent does not move outside of the maze
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        return actions
# defines the experience replay object for our reinforcement learning
class Experience(object):
    # establishes max memory as 100 and discount factor as .97 unless
    # otherwise assigned
    def __init__(self, model, max_memory=100, discount=0.97):
        # initializes our neural network model
        self.model = model
        # initializes maze memory size
        self.max_memory = max_memory
        # initializes discount factor
        self.discount = discount
        # initializes memory list
        self.memory = list()
        # initializes the number of possible actions
        self.num_actions = model.output_shape[-1]
    # defines how each episode of our game is remembered
    # episode = [env_state, action, reward, next_env_state, game_over]
    # memory[i] = episode
    # env_state == flattened 1d maze cells info, including agent cell (see method: observe)
    def remember(self, episode):
        # appends each episode to our memory
        self.memory.append(episode)
        # if our memory exceeds the max memory, remove the first memory
        if len(self.memory) > self.max_memory:
            del self.memory[0]
    # defines how we predict q values from our environment states
    def predict(self, env_state):
        return self.model.predict(env_state)[0]
    # defines how we get data from our game
    def get_data(self, data_size=10):
        # env_state 1d size (1st element of episode)
        env_size = self.memory[0][0].shape[1]
        # initializes length of our memory list
        mem_size = len(self.memory)
        # initializes data size as minimum of memory size or default size of 10
        data_size = min(mem_size, data_size)
        # creates np array of zeros data_size x env_size
        inputs = np.zeros((data_size, env_size))
        # creates np array of zeros data_size x num_actions
        targets = np.zeros((data_size, self.num_actions))
        # populates our inputs and targets arrays with information from random
        # selections of our past memory
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            # gets necessary information from seleced memory
            env_state, action, reward, next_env_state, game_over = self.memory[j]
            # the corresponding inputs row gets the environment state data
            inputs[i] = env_state
            # the corresponding targets row gets the q values for actions
            # based on environment state
            targets[i] = self.predict(env_state)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(next_env_state))
            # if the game is over, retrieve the reward for the action
            if game_over:
                targets[i, action] = reward
            else:
                # or else use Bellman equation to calculate the q values
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
# defiens how we train our neural network
class Qtraining(object):
    def __init__(self, model, env, **opt):
        # Nueral Network Model
        self.model = model
        # maze object environment
        self.env = env
         # Number of epochs to run
        self.n_epoch = opt.get('n_epoch', 1000)
         # Max memory for experiences
        self.max_memory = opt.get('max_memory', 4*self.env.maze.size)
        # Data samples from experience replay
        self.data_size = opt.get('data_size', int(0.75*self.env.maze.size))
         # Starting cells for the agent
        self.agent_cells = opt.get('agent_cells', [(0,0)])
        # Keras model weights file
        self.weights_file = opt.get('weights_file', "")
         # Name for saving weights and json files
        self.name = opt.get('name', 'model')
        # initializes win_count as 0
        self.win_count = 0
        # If you want to continue training from a previous model,
        # just supply the h5 file name to weights_file option
        if self.weights_file:
            print("loading weights from file: %s" % (self.weights_file,))
            self.model.load_weights(self.weights_file)
        # Initialize experience replay object
        self.experience = Experience(self.model, max_memory=self.max_memory)
    # defiens how we train our neural network
    def train(self):
        # get the current time
        start_time = datetime.datetime.now()
        # initializes the seconds of training and win count
        self.seconds = 0
        self.win_count = 0
        # goes through each epoch
        for epoch in range(self.n_epoch):
            # initializes epoch number and loss
            self.epoch = epoch
            self.loss = 0.0
            # resets our maze environment
            self.env.reset(self.env._agent,self.env._advs)
            # initializes game_over as false
            game_over = False
            # get initial env_state (1d flattened canvas)
            self.env_state = self.env.observe()
            # initializes number of episodes as 0
            self.n_episodes = 0
            # play the game until it is over
            while not game_over:
                game_over = self.play()
            # get time since commencement of game
            dt = datetime.datetime.now() - start_time
            # get the seconds since commencement of game
            self.seconds = dt.total_seconds()
            # formats the seconds into time
            t = format_time(self.seconds)
            # format for printing out game updates
            fmt = "Epoch: {:3d}/{:d} | Loss: {:.4f} | Episodes: {:4d} | Wins: {:2d} | flags: {:d} | e: {:.3f} | time: {}"
            #print(fmt.format(epoch, self.n_epoch-1, self.loss, self.n_episodes, self.win_count, len(self.env.uncaptured_flags), self.epsilon(), t))
            show_env(env2)
            # establishes criteria for how many wins before training completion
            if self.win_count > 0:
                #print("Completed training at epoch: %d" % (epoch,))
                break
    # defines playing the game
    def play(self):
        # retrieves action
        action = self.action()
        # establishes environment state as previous environment state
        prev_env_state = self.env_state
        # updates environment state, reward, and game status after taking action
        self.env_state, reward, game_status = self.env.act(action)
        # if we have won the game, update win count
        if game_status == 'win':
            self.win_count += 1
            game_over = True
        # keep playing the game until we lose
        elif game_status == 'lose':
            game_over = True
        else:
            game_over = False
        # if the game is over, update results from our best run if it is the
        # best run so far
        if game_over:
            if self.env.total_reward > self.env.best_total_reward:
                self.env.best_agent_visited = self.env.agent_visited
                self.env.best_total_reward = self.env.total_reward
                self.env.best_win_count = self.win_count
                self.env.best_actions = self.env.agent_actions

        # Store episode (experience) and update episode count
        episode = [prev_env_state, action, reward, self.env_state, game_over]
        self.experience.remember(episode)
        self.n_episodes += 1

        # Train model after getting data from experience replay
        inputs, targets = self.experience.get_data(data_size=self.data_size)
        epochs = int(self.env.base)
        h = self.model.fit(
            inputs,
            targets,
            epochs = epochs,
            batch_size=16,
            verbose=0,
        )
        # calculate loss
        self.loss = self.model.evaluate(inputs, targets, verbose=0)
        return game_over
    # defines actions in the game
    def action(self):
        # Get next action if there are valid actions available
        valid_actions = self.env.agent_valid_actions(self.env.agent)
#        if not valid_actions:
#            action = None
        # choose random action or best action(exploitation vs exploration)
        # based on epsilon value
        if np.random.rand() < self.epsilon():
            # if exploration, choose random action
            action = random.choice(valid_actions)
        else:
            # if exploitation, choose a valid action with highest q value
            # calculate q values for state-action pairs
            q = self.experience.predict(self.env_state)
            # turns q values into dictionary
            q = dict(enumerate(q))
            # only look at q values for valid actions
            q = { key: q[key] for key in qt.env.agent_valid_actions(qt.env.agent)}
            # retrieve valid action with highest q value
            action = max(q, key=q.get)
        return action
    # defines epsilon value
    def epsilon(self):
        # initializes win count
        n = self.win_count
        top = 0.8
        bottom = 0.08
        # if we have won less than 10 times, higher epsilon value
        # decreases as we win more and more
        if n<10:
            e = bottom + (top - bottom) / (1 + 0.1 * n**0.5)
        else:
            e = bottom
        return e
    # Save trained model weights and architecture, this will be used by the visualization code
    def save(self, name=""):
        if not name:
            name = self.name
        h5file = 'model_%s.h5' % (name,)
        json_file = 'model_%s.json' % (name,)
        self.model.save_weights(h5file, overwrite=True)
        with open(json_file, "w") as outfile:
            json.dump(self.model.to_json(), outfile)
        t = format_time(self.seconds)
        print('files: %s, %s' % (h5file, json_file))
        print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (self.epoch, self.max_memory, self.data_size, t))
# builds neural network with 1 input layer, 1 hidden layer, 1 output layer,
# and two activation layers
def build_model(env, **opt):
    loss = opt.get('loss', 'mse')
    a = opt.get('alpha', 0.24)
    model = Sequential()
    esize = env.maze.size
    model.add(Dense(esize, input_shape=(esize,)))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(esize))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(num_actions))
    # uses adam as optimizer and calculates loss from mse
    model.compile(optimizer='adam', loss='mse')
    return model
# builds neural network with 1 intput layer, 2 hidden layers, 3 activation layers,
# and 1 output layer
def build_deepQmodel(env, **opt):
    loss = opt.get('loss', 'mse')
    a = opt.get('alpha', 0.24)
    model = Sequential()
    esize = env.maze.size
    model.add(Dense(esize, input_shape=(esize,)))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(esize))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(esize))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(esize))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(esize))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(num_actions))
    # uses adam as optimizer and calculates loss from mse
    model.compile(optimizer='adam', loss='mse')
    return model
# creates dueling neural network model
def build_dueling_model(env, **opt):
    loss = opt.get('loss', 'mse')
    a = opt.get('alpha', 0.24)
    model = Sequential()
    esize = env.maze.size
    inp = Input(shape=(esize,))
    layer_shared1 = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(inp)
    layer_shared2 = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared1)
    print("Shared layers initialized....")

    layer_v2 = Dense(1,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_shared2)
    layer_a2 = Dense(env.action_space.n,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_shared2)
    print("Value and Advantage Layers initialised....")

    layer_mean = Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))(layer_a2)
    temp = layer_v2
    temp2 = layer_mean

    for i in range(env.action_space.n-1):
        layer_v2 = keras.layers.concatenate([layer_v2,temp],axis=-1)
        layer_mean = keras.layers.concatenate([layer_mean,temp2],axis=-1)
        #print(layer_v2.shape)
        #print(layer_mean.shape)

    layer_q = Subtract()([layer_a2,layer_mean])
    layer_q = Add()([layer_q,layer_v2])

    print("Q-function layer initialized.... :)\n")

    model = Model(inp, layer_q)
    #model.summary()

    model.compile(optimizer='adam', loss='mse')

    return model

# code to show the environment in its current state
def show_env(env, fname=None):
    plt.grid('on')
    n = env.maze.shape[0]
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, n, 1))
    ax.set_yticks(np.arange(0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(env.maze)
    for cell in env.agent_visited:
        if env.agent_visited[cell]:
            canvas[cell] = agent_visited_mark
    for cell in env.flags:
        canvas[cell] = flag_mark
    (agent_row, agent_col), tmp = env.state
    canvas[agent_row, agent_col] = 0.3   # agent cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray', vmin=0, vmax=1)
    if fname:
        plt.savefig(fname)
    return img
# creates heatmap based on agent path
def show_agent_heatmap(env, fname=None):
    plt.grid('on')
    n = env.maze.shape[0]
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, n, 1))
    ax.set_yticks(np.arange(0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(env.maze)
    for cell in agent_heatmap:
        canvas[cell] = agent_heatmap[cell]
    img1 = plt.imshow(canvas, interpolation='none', cmap='gray', vmin=0, vmax=1)
    plt.title(fname)
    if fname:
        plt.savefig(fname)
    return img1
# formats time from seconds
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

if __name__ == '__main__':
    start_time = time.time()
    # initializes dictionaries to track information from best wins of each
    # run
    best_agent_visited = {}
    best_win_count = {}
    best_total_reward = {}
    best_agent_actions = {}
    best_epoch = {}
    # size of the maze: maze_size*maze_size
    maze_size = 9
    # for loop to choose number of runs
    for z in range(50):
        # creates numpy array maze
        maze = np.ones((maze_size,maze_size))
        # initializes flags and adversary positions
        flags = [(1,7),(7,1)]
        advs = [(0,6),(6,0)]
        # builds maze environment object
        env2 = Tmaze(maze, flags, advs)
        # initializes model for environment
        model = build_model(env2)
        # initializes our neural network for q-training
        qt = Qtraining(
            model,
            env2,
            n_epoch = 1000,
            max_memory = 500,
            data_size = 100,
            name = 'model_1'
        )
        # train our neural network
        qt.train()
        # updates our target cell as having been visited
#        if env2.best_agent_visited[(4,4)] == 0:
#            env2.best_agent_visited[(4,4)] = 1
        # updates dictionaries with best win of each run
        best_agent_visited[z] = env2.best_agent_visited
        best_win_count[z] = env2.best_win_count
        best_total_reward[z] = env2.best_total_reward
        best_agent_actions[z] = env2.best_actions
        best_epoch[z] = qt.epoch
    # print out actions from each best run
    #print(best_agent_actions)
    # print out average best reward from runs
    print("average reward:",sum(best_total_reward.values())/len(best_total_reward))
    # prints out average best regret from runs
    print("average regret:",294-sum(best_total_reward.values())/len(best_total_reward))
    # print out epoch
    print("average epoch:",sum(best_epoch.values())/len(best_epoch))
    # print out average best reward from runs
    print("all_win_count",best_win_count)
    # prints out average best regret from runs
    print("all_total_reward",best_total_reward)
    # print out epoch
    print("all_epoch",sum(best_epoch.values())/len(best_epoch))
    print("time:",time.time()-start_time)

    agent_heatmap = {}

    for j in range(0,maze_size):
        for k in range(0,maze_size):
            agent_heatmap[(j,k)] = 1

    # prints out agent path from best win of each run
    for z in range(len(best_win_count)):
        # colors more visited cells darker based on frequency
        for cell in agent_heatmap:
            if best_agent_visited[z][cell] > 0:
                agent_heatmap[cell] -= best_agent_visited[z][cell]*.1

    show_agent_heatmap(env2)
