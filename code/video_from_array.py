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

#game = 'PongDeterministic-v4'
game = 'Pong-v0'
aws_run=True
aws_location='2019-03-21/1231'

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
                    'N_memory':400000,
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





deepQ = DQN.deepQ(game, HYPERPARAMS, PARAMS)
if aws_run:
    deepQ.mp4_from_array('../aws_runs/'+aws_location)
else:
    deepQ.mp4_from_array()
