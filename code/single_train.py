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
