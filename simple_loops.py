import numpy as np
import time

import gym

import DQN

import matplotlib #.pyplot as plt
matplotlib.use('TkAgg') # this makes the fgire in focus rather than temrinal
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import seaborn as sns
#import logging
#logging.getLogger().setLevel(logging.INFO)

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
#game = 'PongNoFrameskip-v4'
game = 'PongDeterministic-v4'
#game = 'Pong-v0'

# ------------------------------------------------------------------------------

# optional bit of code to check the env working visually

env=gym.make(game)

env.reset()

frame, reward, done, tmp = env.step(env.action_space.sample())

#print('\n   frame shape is  -----  ', np.shape(frame), '   ------- \n')

N_obs = np.size(frame,0)


o1 = int( ( (84-8)/4 ) + 1 )
#print(o1)
o2 = int( ( (o1-4)/2 ) + 1 )
#print(o2)
N_squash = o2

#

# 0.001 ok ish

# HYPERPARAMS = {
#                 'ALPHA':2.5e-4,
#                 'GAMMA': 0.99,
#                 'EPSILON_H':1.00,
#                 'EPSILON_L':0.02,
#                 'EPS_DECAY':80000.0,
#                 'EPI_SWAP':10000,
#                 'EPI_START':30,
#                 'N_FILTER':16,
#                 'N_FC':256,
#                 'N_memory':250000,
#                 'N_batch':32,
#                 'UPDATE_FREQ':5000,
#                 'TERMINAL_POINTS':True,
#                 'RATE_INCREASE':1,
#                 'LOSS_SCALE':2.0
#                 }


HYPERPARAMS = {
                'ALPHA':1.0e-4,
                'GAMMA': 0.99,
                'EPSILON_H':1.00,
                'EPSILON_L':0.02,
                'EPS_DECAY':60000.0,
                'EPI_SWAP':10000,
                'EPI_START':4,
                'N_FILTER':1,
                'N_FC':2,
                'N_memory':80000,
                'N_batch':2,
                'UPDATE_FREQ':500,
                'TERMINAL_POINTS':True,
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



N_episodes = 300
results = []
alpha_vec = np.array([1.0e-6,1.0e-4,1.0e-2])
update_vec = np.array([5000,10000])
batch_vec = np.array([32,64,128])
loss_scale_vec = np.array([1.0,2.0,4.0,10.0])
decay_vec = np.array([5.0e3]) #,1.0e4,2.0e4])
rate_inc_vec = np.array([2,4,6])

do_alpha=False

run_type = 'update_freq'

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



    deepQ = DQN.deepQ(game, HYPERPARAMS, PARAMS)
    tmp_dict = deepQ.train(N_episodes)
    #deepQ.game(1)

    # results.append(deepQ.train(N_episodes))


# OUTPUT_STEP = PARAMS['OUTPUT_STEP']
# ep_vec=OUTPUT_STEP*(1+np.arange(int(N_episodes/OUTPUT_STEP) ) )
#
# cols = matplotlib.cm.jet(np.linspace(0,1,len(vals)))
#
# fig,axes = plt.subplots(2,2)
# for i in np.arange(len(vals)):
#     print(results[i]['steps'])
#     axes[0,0].plot(ep_vec,results[i]['rewards'],color=cols[i],label = label0+str(vals[i]))
#     axes[0,0].set_ylabel('avg reward')
#     axes[0,0].set_xlim([0,N_episodes])
#
#     axes[0,1].plot(ep_vec,0.5*(results[i]['maxQ']+results[i]['minQ']),color=cols[i],label = label0+str(vals[i]))
#     axes[0,1].set_ylabel('avg Q')
#     axes[0,1].set_xlim([0,N_episodes])
#
#     axes[1,0].plot(ep_vec,results[i]['actions'],color=cols[i],label = label0+str(vals[i]))
#     #axes[1,0].plot(ep_vec,results[i]['epsilon'],'k',label = label0+str(vals[i]))
#     axes[1,0].set_ylabel('avg action')
#     axes[1,0].set_xlim([0,N_episodes])
#     #axes[1,0].set_ylim([0,1])
#
#     axes[1,1].plot(ep_vec,results[i]['losses'],color=cols[i],label = label0+str(vals[i]))
#     axes[1,1].set_ylabel('avg loss')
#     axes[1,1].set_xlim([0,N_episodes])
#
# plt.legend(frameon=False)
# plt.tight_layout()
# plt.show()

    # plt.plot(ep_vec, results[0]['epsilon'],'k')
    # plt.show()
