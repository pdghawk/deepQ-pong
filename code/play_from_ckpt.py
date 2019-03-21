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

import matplotlib #.pyplot as plt
matplotlib.use('TkAgg') # this makes the fgire in focus rather than temrinal
import matplotlib.pyplot as plt


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

game = 'PongDeterministic-v4'
aws_run = False
aws_location='2019-03-09/1602'

# ------------------------------------------------------------------------------

if aws_run:
    N_episodes = 300
    HYPERPARAMS = {
                    'ALPHA':1.0e-3,
                    'GAMMA': 0.99,
                    'EPSILON_H':1.00,
                    'EPSILON_L':0.02,
                    'EPS_DECAY':70000.0,
                    'EPI_START':20,
                    'N_FILTER':16,
                    'N_FC':256,
                    'N_memory':250000,
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


deepQ = DQN.deepQ(game, HYPERPARAMS, PARAMS)

deepQ.save_game_array()
