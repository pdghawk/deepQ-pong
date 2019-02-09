#import tensorflow as tf
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

#print(N_squash)

#env.render()
#
# frame, reward, done, tmp = env.step(env.action_space.sample())
# print(np.shape(frame))
print(6//2)
run_random_game = True
if run_random_game:
    frame = env.reset()
    ims = []
    tot_reward = 0
    losses = 0
    wins   = 0

    print('action set space size = ',env.action_space.n)
    fig = plt.figure()
    plt.imshow(frame)
    plt.show()
    for i in np.arange(2000):
        Q=np.array([0,0,1])
        action = np.argmax(Q)+1
        frame, reward, done, info = env.step(action) #env.step(env.action_space.sample())
        #env.render()
        #img = env.render(mode='rgb_array')
        #env.close()

        # print(reward, done)
        # print(info['ale.lives'])

        tot_reward+=reward
        if reward==-1.0:
             losses+=1
        if reward==1.0:
            wins+=1
        #     plt.show()

        plt.imshow(frame)
        plt.show()
        #img_list.append(frame.astype(np.uint8))

        # im = plt.imshow(frame, animated=True)
        # ims.append([im])

        #plt.show()
        # print(frame)
        # print(reward)
        # print(done)
        if (done):
            print('ending after', i ,'moves')
            break
    env.close()

    print('losses = ', losses)
    print('wins   = ', wins)


    # plt.imshow(img_list[0])
    # plt.show()
    #
    #
    # frame_out = np.zeros((84,84),dtype=np.uint8)
    # tmp = np.mean(frame, axis=2)
    # tmp = tmp[28:-12, :]
    # tmp = tmp[1:-1:2,::2]
    # frame_out[:,2:-2] = tmp.astype(np.uint8)
    #
    # plt.imshow(frame_out)
    # plt.colorbar()
    # plt.show()



    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
    #
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #
    # ani.save('random_game.mp4',writer=writer)



    #plt.show()

# ------------------------------------------------------------------------------



# 0.001 ok ish
HYPERPARAMS = {
                'ALPHA':1.0e-4,
                'GAMMA': 0.99,
                'EPSILON_H':1.00,
                'EPSILON_L':0.02,
                'EPS_DECAY':60000.0,
                'EPI_SWAP':10000,
                'EPI_START':12,
                'N_FILTER':2,
                'N_FC':4,
                'N_memory':80000,
                'N_batch':2,
                'UPDATE_FREQ':2,
                'RATE_INCREASE':1,
                'LOSS_SCALE':2.0
                }

PARAMS = {  'N_x': 84,
            'N_y': 84,
            'Nc': 4,
            'N_squash':N_squash,
            'OUTPUT_STEP': 4,
            'MAX_STEPS': 2000
            }


if 1==0:

    deepQ = DQN.deepQ(game, HYPERPARAMS, PARAMS)
    #deepQ.game(1)
    N_episodes = 6000
    out_dict = deepQ.train(N_episodes)

    OUTPUT_STEP = 100

    ep_vec=OUTPUT_STEP*(1++np.arange(int(N_episodes/OUTPUT_STEP) - 1) )

    #plt.plot(np.arange(N_episodes),np.asarray(out_dict['losses']),'k')
    #plt.plot(np.arange(N_episodes),np.asarray(out_dict['steps']),'r',label= 'steps used')
    plt.plot(ep_vec,np.asarray(out_dict['steps']),'k',label = 'score')
    plt.plot(ep_vec,np.asarray(out_dict['maxQ']),'b',label = 'avg max Q')
    plt.plot(ep_vec,np.asarray(out_dict['minQ']),'r',label = 'avg min Q')

    plt.legend(frameon=False)

    plt.xlabel('episodes')
    plt.show()

    plt.plot(ep_vec,np.asarray(out_dict['losses']),'k',label = 'avg loss')
    plt.legend(frameon=False)

    plt.xlabel('episodes')

    plt.show()

    plt.plot(ep_vec,np.asarray(out_dict['actions']),'k',label = 'avg action')

    plt.ylim([0, 1])

    plt.xlabel('episodes')
    plt.legend(frameon=False)
    plt.show()

# plt.plot(np.arange(N_episodes),np.asarray(out_dict['Qinit']),'k',label = 'Qinit')
# plt.legend(frameon=False)
# plt.show()

if 1==1:

    N_episodes = 20
    results = []
    alpha_vec = np.array([1.0e-6,1.0e-4,1.0e-2])
    update_vec = np.array([1,3,5])
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
        #deepQ.game(1)

        results.append(deepQ.train(N_episodes))

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

    # plt.plot(ep_vec, results[0]['epsilon'],'k')
    # plt.show()
