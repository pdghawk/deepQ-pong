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

import matplotlib
matplotlib.use('TkAgg') # this makes the fgire in focus rather than temrinal
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
# choose the type of game to play
game = 'Pong-v0'

# will this be running on an aws instance
aws_run = False # True

# plot the results to screen (tensorboard will run regardless)
plot_results=True

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

#
if aws_run:
    N_episodes = 200
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




# ------------------------------------------------------------------------------

# set up a series of hyperparameter scans

# A better way to to do this would be a grid search over all hyperparams (or those
# suspected to be most important), or a random search, which will often outperform
# a grid search.

results = []
alpha_vec = np.array([1.0e-6,1.0e-4,1.0e-2])
update_vec = np.array([1000,5000,10000])
batch_vec = np.array([32,64,128])
loss_scale_vec = np.array([1.0,2.0,4.0,10.0])
decay_vec = np.array([5.0e3]) #,1.0e4,2.0e4])
rate_inc_vec = np.array([2,4,6])

# select which scan you want to run

run_type = 'update_freq'

# set variables according to choice of hyperparameter to scan

if run_type=='alpha':
    vals = alpha_vec
    label0 = 'alpha = '
elif run_type=='update_freq':
    vals = update_vec
    label0 = 'update freq = '
elif run_type=='batch':
    vals = batch_vec
    label0 = 'batch size = '
elif run_type=='loss_scale':
    vals = loss_scale_vec
    label0 = 'loss scale = '
elif run_type=='decay':
    vals = decay_vec
    label0 = 'decay scale = '
elif run_type=='rate_increase':
    vals = rate_inc_vec
    label0 = 'rate_increase = '
else:
    print('Unknown_run_type')

# for each value in the hyperparameter scan, reset the hyperparameter dictionary

for i in np.arange(len(vals)):
    if run_type=='alpha':
        HYPERPARAMS['ALPHA'] = vals[i]
    elif run_type=='update_freq':
        print(' \n ---- running update option  ----- \n')
        #HYPERPARAMS['ALPHA'] = 1.0e-4
        HYPERPARAMS['UPDATE_FREQ'] = vals[i]
    elif run_type=='batch':
        print(' \n ---- running batch option   ----- \n')
        HYPERPARAMS['N_batch'] = vals[i]
    elif run_type=='loss_scale':
        print(' \n ---- running loss option   ----- \n')
        HYPERPARAMS['LOSS_SCALE'] = vals[i]
    elif run_type=='decay':
        print(' \n ---- running decay option')
        HYPERPARAMS['EPS_DECAY'] = vals[i]
    elif run_type=='rate_increase   ----- \n':
        print(' \n ---- running rate increase option')
        HYPERPARAMS['RATE_INCREASE'] = vals[i]
    else:
        print('Unknown run_type')

    # create a deepQ object, i.e set up a deepQ-learning agent
    deepQ = DQN.deepQ(game, HYPERPARAMS, PARAMS)
    # train the model
    tmp_dict = deepQ.train(N_episodes)
    # append the results of the training to results
    results.append(tmp_dict)

# optionally plot the results of the scan.

if plot_results:
    OUTPUT_STEP = PARAMS['OUTPUT_STEP']
    ep_vec=OUTPUT_STEP*(1+np.arange(int(N_episodes/OUTPUT_STEP) ) )

    cols = matplotlib.cm.jet(np.linspace(0,1,len(vals)))

    fig,axes = plt.subplots(2,2)
    for i in np.arange(len(vals)):
        print(results[i]['steps'])
        axes[0,0].plot(ep_vec,results[i]['rewards'],color=cols[i],label = label0+str(vals[i]))
        axes[0,0].set_ylabel('avg reward')
        axes[0,0].set_xlim([0,N_episodes])

        axes[0,1].plot(ep_vec,0.5*(results[i]['maxQ']+results[i]['minQ']),color=cols[i],label = label0+str(vals[i]))
        axes[0,1].set_ylabel('avg Q')
        axes[0,1].set_xlim([0,N_episodes])

        axes[1,0].plot(ep_vec,results[i]['actions'],color=cols[i],label = label0+str(vals[i]))
        #axes[1,0].plot(ep_vec,results[i]['epsilon'],'k',label = label0+str(vals[i]))
        axes[1,0].set_ylabel('avg action')
        axes[1,0].set_xlim([0,N_episodes])
        #axes[1,0].set_ylim([0,1])

        axes[1,1].plot(ep_vec,results[i]['losses'],color=cols[i],label = label0+str(vals[i]))
        axes[1,1].set_ylabel('avg loss')
        axes[1,1].set_xlim([0,N_episodes])

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
