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
import time

import gym

import DQN

import matplotlib #.pyplot as plt
matplotlib.use('TkAgg') # this makes the fgire in focus rather than temrinal
import matplotlib.pyplot as plt


import seaborn as sns

# ------------------------------------------------------------------------------

matplotlib.rcParams['lines.linewidth']  = 1.5
matplotlib.rcParams['axes.linewidth']   = 1.5
matplotlib.rcParams['font.weight']      = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['font.size']        = 10
matplotlib.rcParams['legend.frameon']   = False
matplotlib.rcParams['figure.figsize']   = (22/2.54, 15/2.54)
#sns.set()
# ------------------------------------------------------------------------------

game = 'Pong-v0'

# ------------------------------------------------------------------------------

# optional bit of code to check the env working visually


env=gym.make(game)

env.reset()

frame, reward, done, tmp = env.step(env.action_space.sample())

print('\n   frame shape is  -----  ', np.shape(frame), '   ------- \n')

N_obs = np.size(frame,0)


o1 = int( ( (84-8)/4 ) + 1 )
print(o1)
o2 = int( ( (o1-4)/2 ) + 1 )
print(o2)
N_squash = o2

print(N_squash)

# ------------------------------------------------------------------------------

HYPERPARAMS = {
                'ALPHA':2.5e-4,
                'GAMMA': 0.99,
                'EPSILON_H':1.00,
                'EPSILON_L':0.02,
                'EPS_DECAY':80000.0,
                'EPI_SWAP':10000,
                'EPI_START':30,
                'N_FILTER':16,
                'N_FC':256,
                'N_memory':250000,
                'N_batch':32,
                'UPDATE_FREQ':7000,
                'RATE_INCREASE':1,
                'LOSS_SCALE':2.0
                }

PARAMS = {  'N_x': 84,
            'N_y': 84,
            'Nc': 4,
            'N_squash':N_squash,
            'OUTPUT_STEP': 5,
            'MAX_STEPS': 20000
            }


deepQ = DQN.deepQ(game, HYPERPARAMS, PARAMS)
deepQ.save_animated_game(dir='../aws_runs/2019-02-13/1603')
