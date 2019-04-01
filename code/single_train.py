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
import DQN


# ------------------------------------------------------------------------------

# choose the type of game to play
#game = 'PongNoFrameskip-v4'
#game = 'PongDeterministic-v4'
game = 'Pong-v0'

# will this be running on an aws instance
aws_run = False

# two different options below for entering hyperparameters for when running on
# an aws instance vs local machine. Can set the local machine hyperparamters to
# be smaller for easier testing

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
    N_episodes = 400
    HYPERPARAMS = {
                    'ALPHA':3.0e-4,
                    'GAMMA': 0.99,
                    'EPSILON_H':1.00,
                    'EPSILON_L':0.03,
                    'EPS_DECAY':80000.0,
                    'EPI_START':40,
                    'N_FILTER':32,
                    'N_FC':512,
                    'N_memory':400000,
                    'N_batch':32,
                    'UPDATE_FREQ':5000,
                    'TERMINAL_POINTS':True,
                    'LOSS_SCALE':2.0
                    }
    PARAMS = {  'Nc': 4,
                'OUTPUT_STEP': 10,
                'MAX_STEPS': 20000
                }
else:
    # we are using the local machine - small hyperparams for testing
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



# create a deepQ object, i.e set up a deepQ-learning agent
deepQ = DQN.deepQ(game, HYPERPARAMS, PARAMS)
# train that agent
tmp_dict = deepQ.train(N_episodes)
# one can observe the training in tensorboard.
# tensorboard summaries are logged in deepQ-pong/data_summaries
# in navigate to deepQ-pong in terminal and run: tensorboard --logdir=data_summaries
