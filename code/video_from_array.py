# Copyright 2019 Peter Hawkins
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import numpy as np

import matplotlib #.pyplot as plt
matplotlib.use('Agg') # this makes the fgire in focus rather than temrinal
import matplotlib.pyplot as plt

import DQN

# ------------------------------------------------------------------------------

# choose the type of game to play
#game = 'PongDeterministic-v4'
game = 'Pong-v0'

# will this be running on an aws instance, or obtaining data from an aws run
aws_run=True

# where should the code look for game array data from which to create a video
aws_location='2019-03-21/1650'
# ------------------------------------------------------------------------------

# We will set up dictionaries of Hyperparameters (HYPERPARAMS), and parameters
# (PARAMS) of the model to be passed to the DQN.

# See the deepQ-pong documentation, and in particular the DQN package docs, for
# further details.

# HYPERPARAMS should consist of:
 # - 'ALPHA': learning rate
 # - 'GAMMA': reward discount factor
 # - 'EPSILON_H': initial probability of random actions in training
 # - 'EPSILON_L': lowest probability of random actions in training
 # - 'EPS_DECAY': decay rate (units of frames) of epsilon (exp(-frame/EPS_DECAY))
 # - 'EPI_START': episode at which to begin training
 # - 'N_FILTER': Number of filters for initial convolutional layer
 # - 'N_FC': Number of hidden units in fully connected layer
 # - 'N_memory': Number of transitions to store
 # - 'N_batch': The mini-batch size
 # - 'UPDATE_FREQ': how many frames to train on between updates of target network
 # - 'TERMINAL_POINTS': count a single point loss as a terminal move (boolean)
 # - 'LOSS_SCALE': scale on Huber loss, for testing, keep as 2.0

# PARAMS should consist of:
 # - 'Nc': number of frames in a single game state
 # - 'OUTPUT_STEP': How often (in episodes) to save output summaries
 # - 'MAX_STEPS': max number of frames allowed per episode

# ------------------------------------------------------------------------------
if aws_run:
    N_episodes = 300
    HYPERPARAMS = {
                    'ALPHA':3.0e-4,
                    'GAMMA': 0.99,
                    'EPSILON_H':1.00,
                    'EPSILON_L':0.02,
                    'EPS_DECAY':80000.0,
                    'EPI_START':40,
                    'N_FILTER':32,
                    'N_FC':512,
                    'N_memory':500000,
                    'N_batch':32,
                    'UPDATE_FREQ':5000,
                    'TERMINAL_POINTS':True,
                    'LOSS_SCALE':2.0
                    }
    PARAMS = {  'Nc': 4,
                'OUTPUT_STEP': 2,
                'MAX_STEPS': 20000
                }
else:
    N_episodes = 10
    HYPERPARAMS = {
                    'ALPHA':1.5e-4,
                    'GAMMA': 0.99,
                    'EPSILON_H':1.00,
                    'EPSILON_L':0.02,
                    'EPS_DECAY':60000.0,
                    'EPI_START':4,
                    'N_FILTER':1,
                    'N_FC':2,
                    'N_memory':80000,
                    'N_batch':4,
                    'UPDATE_FREQ':5000,
                    'TERMINAL_POINTS':True,
                    'LOSS_SCALE':2.0
                    }

    PARAMS = {  'Nc': 4,
                'OUTPUT_STEP': 2,
                'MAX_STEPS': 20000
                }




# create a deepQ object, i.e set up a deepQ-learning agent, the hyperparameters
# should match a game already trained with single_train.py or equivilent code, and
# for which an array representing the gameplay has been created with play_from_ckpt.py
deepQ = DQN.deepQ(game, HYPERPARAMS, PARAMS)

# create the video from the game array, game array will load automatically according
# to the hyperparameters specified.
if aws_run:
    # look in aws_location for the files
    deepQ.mp4_from_array('../aws_runs/'+aws_location)
else:
    # use the defualt search path ('..')
    deepQ.mp4_from_array()
