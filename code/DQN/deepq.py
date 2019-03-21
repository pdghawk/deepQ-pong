import tensorflow as tf
from tensorflow import keras
#from keras import backend as K
import numpy as np
import time
import os

import matplotlib #.pyplot as plt
matplotlib.use('TkAgg') # this makes the fgire in focus rather than temrinal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gym

from DQN.qmemory import Qmemory

class deepQ:
    """ Object for deep Q learning, for solving openai gym environments

    deep Q network can be used for Q learning, to find the Q function that maximses
    the reward, and effectively therefore gives an optimal stragey for the game.

    The method used here contains various elements of the deepQ algorithm, namely:
    experience replay, double Q learning, online and target networks with strided updates
    """

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def __init__(self,game,HYPERPARAMS,PARAMS):
        """ Initialize

        Initialize the hyperparameters of the model, start the game environment,
        setup the tensorflow graph, start a filenaming convention for results.

        Args:
            HYPERPARAMS: a dictionary of hyperparameters:
                - ALPHA: learning rate
                - GAMMA: reward discount factor
                - EPSILON_H: initial probability of random actions in training
                - EPSILON_L: lowest probability of random actions in training
                - EPS_DECAY: decay rate (units of frames) of epsilon (exp(-frame/EPS_DECAY))
                - EPI_START: episode at which to begin training
                - N_FILTER: Number of filters for initial convolutional layer
                - N_FC: Number of hidden units in fully connected layer
                - N_memory: Number of transitions to store
                - N_batch: The mini-batch size
                - UPDATE_FREQ: how many frames to train on between updates of target network
                - TERMINAL_POINTS: count a single point loss as a terminal move (boolean)
                - LOSS_SCALE: scale on Huber loss, for testing, keep as 2.0
            PARAMS: A dictionary of parameters of the model:
                - N_x: x dimension of a prepprocessed frame (pixels)
                - N_y: y dimension of a prepprocessed frame (pixels)
                - Nc: number of frames in a single game state
                - N_squash: dimensions in x,y of state after both conv layers applied
                - OUTPUT_STEP: How often (in episodes) to save output summaries
                - MAX_STEPS: max number of frames allowed per episode

        """

        self.env = gym.make(game)

        self.HYPERPARAMS = HYPERPARAMS
        self.PARAMS   = PARAMS


        self.N_action = self.env.action_space.n//2
        # ^ this is specific to Pong, becuase half the actions have the same
        # purose as the other actions.
        # e.g actions 2 and 4 both move the paddle up
        # 0: nothing, 1: nothing, 2:up, 3:down, 4:up, 5:down
        # use instead x= 0,1,2, with mapping action = x+1,
        # then x=0: nothing, x=1:up, x=2:down
        # so we reduce the action space by half (from 6 to 3)
        # and consequently action from Q value will be argmax(Q)+1
        # n.b Q is vector of length N_action (i.e now 6//2 = 3)

        # use the environment and model params to find useful quantities
        frame = self.env.reset()
        frame = self.preprocess(frame)
        self.PARAMS['N_x'] = np.size(frame,0)
        self.PARAMS['N_y'] = np.size(frame,1)
        o1 = int( ( (self.PARAMS['N_x']-8)/4 ) + 1 )
        o2 = int( ( (o1-4)/2 ) + 1 )
        o3 = int( ( (o2-3)/1) + 1)
        self.PARAMS['N_squash'] = o3

        tf.reset_default_graph()
        self.graph = tf.get_default_graph() #tf.Graph()

        alpha_txt = f"alpha_{HYPERPARAMS['ALPHA']:.2e}_"
        upd_txt   = f"updfreq_{HYPERPARAMS['UPDATE_FREQ']:d}_"
        decay_txt = f"EPSDECAY_{HYPERPARAMS['EPS_DECAY']:.1f}_"
        nfc_txt   = f"NFC_{HYPERPARAMS['N_FC']:d}_"
        nfilt_txt = f"Nfilter_{HYPERPARAMS['N_FILTER']:d}_"
        mem_txt   = f"mem_{HYPERPARAMS['N_memory']:d}_"
        batch_txt = f"batch_{HYPERPARAMS['N_batch']:d}_"
        term_txt  = f"terminal_{HYPERPARAMS['TERMINAL_POINTS']:d}"
        self.params_text = alpha_txt+upd_txt+decay_txt+nfc_txt+\
                           nfilt_txt+mem_txt+batch_txt+term_txt

        print("\n==========================================================")
        print("\n\n filename for saving       : ",self.params_text)
        print(" action space has size     : ",self.N_action)
        print(" using tensorflow version  : ",tf.VERSION, "\n\n")
        print("==========================================================\n")
        return None

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------


    def preprocess(self,frame):
        """ Preprocess frame of game

        Args:
            frame: a frame of the game

        Returns:
            frame_out: the preprocessed frame
        """
        frame_out = np.zeros((84,84),dtype=np.uint8)
        # to black and white
        tmp = np.mean(frame, axis=2)
        # trim edges
        tmp = tmp[28:-12, :]
        # downsample
        tmp = tmp[1:-1:2,::2]
        frame_out[:,2:-2] = tmp.astype(np.uint8)
        return frame_out

    def action2step(self,act):
        """ Convert integer into game action

        In order that Pong can have only 3 actions (nothing, up, down), rather
        than the 6 (each action replicated) in the gym environment, use a
        preprocessing for the actions.

        Args:
            act: integer representing an action
        Returns:
            step: an integer for the action, act, expected by the game

        """
        step=act+1
        return step

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    def Qnet(self,obs,call_type,trainme,reuseme):
        """ Neural network to get Q for given state

        Structure of the network is:

        - convolutional layer (K=8,S=4) with N_FILTER filters
        - convolutional layer (K=4,S=2) with 2*N_FILTER filters
        - convolutional layer (K=3,S=1) with 2*N_FILTER filters
        - Fully Connected layer with N_FC hidden units

        It takes in input observation (a state of a game), and returns the predicted
        value Q for this action. The maximal position within Q is the policy action.

        Args:
            obs: (tensor) set of observations to predict Q for: size: batch,(x,y..),frames
                frames should be 4 to match deepmind paper.
            call_type: 'online/' or 'target/' - which network to use
            trainme: (bool) should the weights be trainable
            reuseme: (bool) should the weights be reusable

        Returns:
            z_out: output of the Neural Net, which is the predicted Q for the observation

        """

        with tf.variable_scope(call_type):
            z = tf.reshape(obs, [-1,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc']])
            #print(z.shape)
            with tf.variable_scope('conv_layer0',reuse=reuseme):
                z_conv0 = tf.layers.Conv2D(filters = self.HYPERPARAMS['N_FILTER'],
                                            kernel_size = (8,8),
                                            strides = (4,4),
                                            padding='valid',
                                            activation=tf.nn.leaky_relu,
                                            trainable=trainme,
                                            kernel_initializer=tf.keras.initializers.he_normal())(z)

            with tf.variable_scope('conv_layer1',reuse=reuseme):
                z_conv1 = tf.layers.Conv2D(filters = 2*self.HYPERPARAMS['N_FILTER'],
                                            kernel_size = (4,4),
                                            strides = (2,2),
                                            padding='valid',
                                            activation=tf.nn.leaky_relu,
                                            trainable=trainme,
                                            kernel_initializer=tf.keras.initializers.he_normal())(z_conv0)
                #z_conv1_flat = tf.reshape(z_conv1,[-1,self.PARAMS['N_squash']*self.PARAMS['N_squash']*(2*self.HYPERPARAMS['N_FILTER'])])

            with tf.variable_scope('conv_layer2',reuse=reuseme):
                z_conv2 = tf.layers.Conv2D(filters = 2*self.HYPERPARAMS['N_FILTER'],
                                            kernel_size = (3,3),
                                            strides = (1,1),
                                            padding='valid',
                                            activation=tf.nn.leaky_relu,
                                            trainable=trainme,
                                            kernel_initializer=tf.keras.initializers.he_normal())(z_conv1)
                z_flat = tf.reshape(z_conv2,[-1,self.PARAMS['N_squash']*self.PARAMS['N_squash']*(2*self.HYPERPARAMS['N_FILTER'])])


            with tf.variable_scope('FC_layer0',reuse=reuseme):
                z_FC0 =  tf.layers.Dense(units=self.HYPERPARAMS['N_FC'],activation=tf.nn.relu,trainable=trainme,kernel_initializer=tf.keras.initializers.he_normal())(z_flat)

            with tf.variable_scope('layer_out',reuse=reuseme):
                z_out = tf.layers.Dense(units=self.N_action,trainable=trainme,kernel_initializer=tf.keras.initializers.he_normal())(z_FC0)

        return z_out

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    def update_layer(self,layer):
        """ Update the weights/biases of target network

        For stability, it is useful to actively train an online network, and
        only periodically update a target network with the weights and biases of
        the online network. This method updates a gicen layer in the target
        network to be the same as the equivilent layer of the online network

        Args:
            layer: (string) name of layer. e.g. 'layer0'

        Returns:
            upd_k: operator that updates the kernel of the layer
            epd_b: operator that updates the bias of the layer
        """

        with tf.name_scope('get_online_wieghts'):
            with tf.variable_scope('online/' + layer,reuse=True):
                k_online = tf.get_variable('kernel')
                b_online = tf.get_variable('bias')

        with tf.name_scope('get_target_weights'):
            with tf.variable_scope('target/' + layer,reuse=True):
                k_target = tf.get_variable('kernel')
                b_target = tf.get_variable('bias')
        with tf.name_scope('assign_new_target_weights'):
            upd_k = tf.assign(k_target,k_online)
            upd_b = tf.assign(b_target,b_online)
        return upd_k,upd_b

    #---------------------------------------------------------------------------

    def make_graph(self):
        """ Define the computational graph

        Takes in the game states (before and after action), action, reward, and
        whether terminal as placeholders. Uses these to compute Q values for
        both online and target networks. Applies the double deep Q learning
        algorithm, using self.Qnet as the neural network which predicts the
        Q values for a given state.


        Placeholders: the follwowing variables should be set with a feed_dict
                - phi_i_: state before action
                - phi_j_: state after action
                - a_i_: action taken
                - r_i_: reward for taking action
                - t_i_: terminal move signifier (0 if final, 1 otherwise)

        Returns:
            graph_vars: dictionary of variables of graph which are useful:
                        - graph_init:       graph initializer (global)
                        - graph_local_init: graph initializer (local)
                        - Q_i_:Q values predicted by Qnet on phi_i (online net)
                        - loss_:    loss on batch,
                        - train_op: training tf op
                        - update_target:updates target network weights to online weights
                        - merged: op to merge summaries for tensorboard
                        - phi_i_: placeholder phi_i_
                        - phi_j_: placeholder phi_j_
                        - a_i_:  placeholder a_i_,
                        - r_i_:  placeholder r_i_,
                        - t_i_:  placeholder t_i_,
                        - saver: tf saver for saving meta graph and variables


        """
        with self.graph.as_default():

            # placeholders for the states, actions, rewards, and whether terminal
            # size is batch, (x, y,), stored frames(4)
            phi_i_ = tf.placeholder(shape=[None,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc']],dtype=tf.float32)
            phi_j_ = tf.placeholder(shape=[None,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc']],dtype=tf.float32)
            a_i_   = tf.placeholder(shape=[None,1],dtype=tf.uint8)
            r_i_   = tf.placeholder(shape=[None,1],dtype=tf.float32)
            t_i_   = tf.placeholder(shape=[None,1],dtype=tf.float32)

            # ------------------------------------------------------------------
            with tf.name_scope('Q_i_online'):
                Q_i_ = self.Qnet(phi_i_,'online',True,False)
                #print("Q_i_ shape         = ",Q_i_.shape)

            with tf.name_scope('Value_function_i_online'):
                # convert actions that were taken into onehot format
                a_list = tf.reshape(tf.cast(a_i_,tf.int32),[-1])
                print("a_list shape = ",a_list.shape)

                a_onehot = tf.one_hot(a_list, self.N_action)
                print(a_onehot.shape)

                # now use the onehot format actions to select the Q_i's that are actually
                # obtained by taking action a_i. n.b Qnet returns a value for Q for all actions
                # but we only want to know Q for the action taken

                V_i_tmp = tf.multiply(a_onehot,Q_i_)
                print(V_i_tmp.shape)
                V_i_ = tf.reduce_sum(V_i_tmp, axis=1)
                print(V_i_.shape)


            # ------------------------------------------------------------------
            # we need to get the actions to take on the Q_target step, by using the expected action
            # that the online network predicts is best, but then use the Q from the target net with the
            # online selected action

            # this is the same network as for Q_i_ - we set reuse=True
            # (it is also trainable)
            with tf.name_scope('Qj_online'):
                Qj_online_ = self.Qnet(phi_j_,'online',True,True)
                Qj_online_inds = tf.argmax(Qj_online_,axis=1)
                Qj_onehot_inds = tf.one_hot(Qj_online_inds, self.N_action)

            # ------------------------------------------------------------------

            # this has reuse=False, make a new network - the target network
            # it is not trainable. Instead we train the online network and
            # set the weights/biases of the layers in the target network to be the
            # same as those in the online network every so many games.
            with tf.name_scope('Qj_target'):
                Q_j_ = self.Qnet(phi_j_,'target',False,False)

            # now only take values of Q (target) for state j, using action that
            # the online network would predict
            with tf.name_scope('value_function_j'):
                V_j_ = tf.reduce_sum(tf.multiply(Qj_onehot_inds,Q_j_),axis=1)

            # ------------------------------------------------------------------
            # get the future discounted reward
            with tf.name_scope('discounted_reward'):
                y_          = tf.add( tf.squeeze(r_i_) , self.HYPERPARAMS['GAMMA']*tf.multiply(tf.squeeze(t_i_),tf.squeeze(V_j_)))

            print("y shape = ",y_.shape)
            print("r_i_ shape = ",tf.squeeze(r_i_).shape)

            # difference between value function (future discounted) and the value
            # funtion on state i
            with tf.name_scope('discount_take_value'):
                x_    = tf.subtract( y_, V_i_  )

            print("x_ shape = ",x_.shape)

            # ------------------------------------------------------------------
            # define the loss, create an optimizer op, and a training op

            # use a Pseudo-Huber loss
            with tf.name_scope('loss'):
                loss_scale = self.HYPERPARAMS['LOSS_SCALE'] # how steep loss is for large values
                loss_ = tf.reduce_mean( loss_scale*(tf.sqrt(1.0+(1.0/loss_scale)**2*tf.multiply(x_,x_)) - 1.0) )

            with tf.name_scope('optimizer'):
                optimizer    = tf.train.RMSPropOptimizer(self.HYPERPARAMS['ALPHA'])
                train_op     = optimizer.minimize(loss_)

            # ------------------------------------------------------------------
            # update the parameters of the target network, by cloning those from
            # online Q network. This will only be sess.run'ed every C steps

            with tf.name_scope('target_updates'):
                upd_c_k0,upd_c_b0   = self.update_layer('conv_layer0/conv2d')
                upd_c_k1,upd_c_b1   = self.update_layer('conv_layer1/conv2d')
                upd_c_k2,upd_c_b2   = self.update_layer('conv_layer2/conv2d')
                upd_FC_k0,upd_FC_b0 = self.update_layer('FC_layer0/dense')
                upd_k_out,upd_b_out = self.update_layer('layer_out/dense')

                # group all of these update ops into a single op for updating the
                # entire target network
                update_target = tf.group(upd_c_k0, upd_c_b0, upd_c_k1, upd_c_b1,upd_c_k2, upd_c_b2, upd_FC_k0, upd_FC_b0, upd_k_out, upd_b_out)

            # ------------------------------------------------------------------
            # create some tenorboard outputs for real-time analysis
            tf.summary.scalar('loss', tf.squeeze(loss_))
            merged = tf.summary.merge_all()
            # ------------------------------------------------------------------

            graph_init = tf.global_variables_initializer()
            graph_local_init = tf.local_variables_initializer()

        saver = tf.train.Saver()

        graph_vars = {'graph_init':graph_init,
                        'graph_local_init':graph_local_init,
                        'Q_i_':Q_i_,
                        'loss_':loss_,
                        'train_op':train_op,
                        'update_target':update_target,
                        'merged':merged,
                        'phi_i_':phi_i_,
                        'phi_j_':phi_j_,
                        'a_i_':a_i_,
                        'r_i_':r_i_,
                        't_i_':t_i_,
                        'saver':saver}
        return graph_vars


    def summary_hist(self,summary_,tag,data,bins):
        """ Add Histogram to tensorboard summary

        Args:
            summary_: a tf summary object to add histogram to
            tag: a name/tag for the histogram
            data: The data to be plotted
            bins: The number of bins for the histogram, or an array of bin edges
        """
        npdata = np.asarray(data)
        hist_vals, bin_edges = np.histogram(npdata,bins)

        hist = tf.HistogramProto()
        hist.min = np.min(npdata)
        hist.max = np.max(npdata)

        bin_edges = bin_edges[:-1]

        for b in bin_edges:
            hist.bucket_limit.append(b)
        for hv in hist_vals:
            hist.bucket.append(hv)

        summary_.value.add(tag=tag,histo=hist)



        return None
    #---------------------------------------------------------------------------

    def train(self, N_episodes):
        """Train the DeepQ network

        Args:
            N_epsiodes: how many episodes to train over

        Returns:
            out_dict: A dictionary of how various things evolved during during:
                    - rewards':  average reward per epsiode
                    - 'steps':   average steps per epsiode
                    - 'maxQ':    average max_Q epsiode
                    - 'minQ':    average min_Q per epsiode
                    - 'losses':  average loss per epsiode
                    - 'actions': average action per epsiode
                    - 'epsilon': average epsilon per epsiode


        """

        # define the computational graph for traing the Q network
        graph_vars = self.make_graph()


        # ----------------------------------------------------------------------
        # ----------------- now use the graph above as the session -------------
        with tf.Session(graph=self.graph) as sess:


            sess.run(graph_vars['graph_init'])
            sess.run(graph_vars['graph_local_init'])

            print(tf.trainable_variables())

            summary=tf.Summary()
            #writer = tf.summary.FileWriter('%s/%s' % ('./../data_summaries', time.strftime("%Y%m%d-%H%M%S")),sess.graph)
            writer = tf.summary.FileWriter('%s/%s' % ('./../data_summaries', self.params_text),sess.graph)


            # ------------------------------------------------------------------
            # arrays in which to store memory of states it has seen

            # CREATE TWO MEMORY TYPES
            # 'normal' memory - stores non-final moves
            # 'losses' memory - stores only final (losing) moves

            # the idea of this is to keep a consistent number of losing/winning
            # and 'normal' moves, so that the number of each type used in training
            # stays consistent

            N_mem_normal = int(0.7*self.HYPERPARAMS['N_memory'])
            N_mem_losses = int(0.15*self.HYPERPARAMS['N_memory']) # - N_mem_normal #int(0.05*self.HYPERPARAMS['N_memory'])
            N_mem_wins   = self.HYPERPARAMS['N_memory'] - N_mem_normal - N_mem_losses

            memory_normal = Qmemory(N_mem_normal,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc'])
            memory_wins   = Qmemory(N_mem_wins  ,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc'])
            memory_losses = Qmemory(N_mem_losses,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc'])

            # also define how big each batch should be
            N_batch_l = int(0.15*self.HYPERPARAMS['N_batch'])
            N_batch_w = N_batch_l
            N_batch_n = self.HYPERPARAMS['N_batch'] - N_batch_w - N_batch_l

            print(N_batch_n,N_batch_l,N_batch_w)
            # ------------------------------------------------------------------

            # counter for number of steps taken
            steps_count = 0

            # initialise arrays for storing average values of quantities
            reward_p_ep = np.zeros((int(N_episodes/self.PARAMS['OUTPUT_STEP']),))
            steps_p_ep  = np.zeros_like(reward_p_ep)
            avQ_p_ep    = np.zeros_like(reward_p_ep)
            max_Q_p_ep  = np.zeros_like(reward_p_ep)
            min_Q_p_ep  = np.zeros_like(reward_p_ep)
            loss_p_ep   = np.zeros_like(reward_p_ep)
            av_action_p_ep = np.zeros_like(reward_p_ep)
            Q_init_0_p_ep = np.zeros_like(reward_p_ep)
            epsilon_ep    = np.zeros_like(reward_p_ep)

            #these just for testing
            init_obs = self.env.reset()
            init_obs = self.preprocess(init_obs)
            init_phi = np.tile( init_obs[:,:,np.newaxis], (1,1,self.PARAMS['Nc']) )

            out_count=0
            # --------------- loop over games ----------------------------------
            time_ep1 = time.time()
            for epi in np.arange(N_episodes):

                # reset the game, and initialise states
                done=False

                current_obs = self.env.reset()
                current_obs = self.preprocess(current_obs)

                current_phi = np.tile( current_obs[:,:,np.newaxis], (1,1,self.PARAMS['Nc']) )

                new_obs= np.zeros_like(current_obs)
                new_phi = np.zeros_like(current_phi)


                # --------------------------------------------------------------
                # define epsilon (chance to make move based on policy vs random)
                # for the fist 'EPI_START' episodes only make random moves
                # after this, decay epsilon exponentially according to total steps
                # taken during training
                if epi<self.HYPERPARAMS['EPI_START']:
                    eps_tmp = self.HYPERPARAMS['EPSILON_H']
                else:
                    eps_tmp = self.HYPERPARAMS['EPSILON_L'] + (self.HYPERPARAMS['EPSILON_H'] - self.HYPERPARAMS['EPSILON_L'])*np.exp(-(steps_count*1.0)/self.HYPERPARAMS['EPS_DECAY'])

                # --------------------------------------------------------------
                # initilaize counters
                tot_reward = 0.0
                steps_used = 0.0

                maxQ = 0.0

                # reset the lists to empty at beginning of new avergaing period
                if ((np.mod(epi,self.PARAMS['OUTPUT_STEP'])==1 and epi>1) or epi==0):
                    losses=[]
                    av_acts=[]
                    maxQs = []
                    minQs = []
                    steps_list=[]
                    reward_list = []

                # ---------- LOOP over frames in a given episode ---------------

                for i in np.arange(self.PARAMS['MAX_STEPS']):
                    # get action using the Q net, or at random

                    if np.random.uniform() < eps_tmp:
                        action = np.asarray(np.random.randint(self.N_action))
                        new_obs, reward, done, info = self.env.step(self.action2step(action))
                    else:
                        # feed data into the session graph to get Q as a numpy array
                        # only phi_i is actual data
                        # phi_j, a_i, r_i are just dummy really as not used

                        tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:]/255.0,
                                        graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:]/255.0,
                                        graph_vars['a_i_']:memory_normal.memory_a_i[:1,:],
                                        graph_vars['r_i_']:memory_normal.memory_r_i[:1,:],
                                        graph_vars['t_i_']:memory_normal.memory_terminal_i[:1,:]}

                        # use Q network graph to get Q_i, uses the online network
                        Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))

                        # append the max and min Q to the lists (will be averaged later)
                        maxQs.append(np.amax(Q))
                        minQs.append(np.amin(Q))

                        # the action to be taken, is one that maximises Q
                        action = np.argmax(Q)

                        new_obs, reward, done, info = self.env.step(self.action2step(action))
                        av_acts.append(action)



                    # ----------------------------------------------------------


                    # preprocess the image
                    new_obs = self.preprocess(new_obs)

                    # phi is made of several observations/frames. so concatenate
                    # the the current phi (all but first frame), with the new observation
                    # this then becomes the new state containg 'Nc' frames
                    new_phi = np.concatenate((current_phi[:,:,1:],new_obs[:,:,np.newaxis]), axis=2)

                    # convert the boolean 'done' which tells us if move is the last
                    # of game, to a float (number). we want it to be 0.0 when it is
                    # the final move - so we need the opposite of normal conversion
                    # of bool->float - so use:   (not done)

                    if not self.HYPERPARAMS['TERMINAL_POINTS']:
                        term_float = np.array(not done)
                        term_float = term_float.astype(np.float32)
                    else:
                        term_float = np.array(reward > -0.1).astype(np.float32)


                    tot_reward+=reward

                    # ----------------------------------------------------------
                    # WRITE new experience to MEMORY

                    # only perform if we are more than 'Nc' moves into this game.
                    # this is because we need to have phi states of length 'Nc'
                    # and they are initialised at beginning of game to be the initial
                    # frame repeated Nc times, which would be unrealistic - so do
                    # not add to memory.
                    if i>=(self.PARAMS['Nc']-1):
                        if reward > 0.1:
                            #print("writing win, ",reward)
                            memory_wins.write(current_phi, new_phi, action, reward, term_float)
                        elif reward > -0.1:
                            #print("writing normal, ",reward)
                            memory_normal.write(current_phi, new_phi, action, reward, term_float)
                        else:
                            #print("writing loss, ",reward)
                            memory_losses.write(current_phi, new_phi, action, reward, term_float)

                    # ----------------------------------------------------------

                    # APPLY LEARING UPDATES

                    # ----------------------------------------------------------

                    # take a batch of the experiences from memory
                    # only do if experience memory is big enough to contain N_batch entries
                    #if (mem_count>self.HYPERPARAMS['N_batch']):
                    if(epi>self.HYPERPARAMS['EPI_START']):

                        # define sizes of batches for the 'normal' memory, and
                        # batch size for the 'losses' memory.


                        batch_n = memory_normal.get_batch(N_batch_n)
                        batch_l = memory_losses.get_batch(N_batch_l)
                        batch_w = memory_losses.get_batch(N_batch_w)

                        # combine the batches from both memories to create a single
                        # batch which represents 'normal' and 'loss' moves with a
                        # predetermined ratio.

                        phi_i_batch = np.concatenate((batch_n['phi_i'], batch_l['phi_i'], batch_w['phi_i'])  , axis=0)/255.0
                        phi_j_batch = np.concatenate((batch_n['phi_j'], batch_l['phi_j'], batch_w['phi_j'] ) , axis=0)/255.0
                        a_i_batch   = np.concatenate((batch_n['a_i']  , batch_l['a_i'], batch_w['a_i'])    , axis=0)
                        r_i_batch   = np.concatenate((batch_n['r_i']  , batch_l['r_i'], batch_w['r_i'])    , axis=0)
                        t_i_batch   = np.concatenate((batch_n['t_i']  , batch_l['t_i'], batch_w['t_i'])    , axis=0)


                        #print(np.shape(phi_i_batch))
                        feed_dict_batch = { graph_vars['phi_i_']:(phi_i_batch).astype(np.float32),
                                            graph_vars['phi_j_']:(phi_j_batch).astype(np.float32),
                                            graph_vars['r_i_']:r_i_batch,
                                            graph_vars['a_i_']:a_i_batch,
                                            graph_vars['t_i_']:t_i_batch}

                        # get the loss for this batch
                        loss0 = sess.run(graph_vars['loss_'],feed_dict=feed_dict_batch)
                        # append loss to be averaged later
                        losses.append(loss0)


                        # APPLY GRADIENT DESCENT for batch
                        # only perform if episopde is > EPI_START
                        # if(epi==self.HYPERPARAMS['EPI_START']):
                        #     graph_vars['train_op'].run(feed_dict=feed_dict_batch,options=run_options,run_metadata=run_metadata)
                        #     sess.run(tmp_loss,graph_vars['train_op'],)

                        if(epi>self.HYPERPARAMS['EPI_START']):
                            graph_vars['train_op'].run(feed_dict=feed_dict_batch)


                    # ----------------------------------------------------------

                    # prepare for beginning a new game, update counters etc

                    # RESET what the current phi is for the next step
                    current_phi = 1.0*new_phi

                    # if we are in the training period - add one to total number
                    # of steps taken total over all episodes
                    if epi>self.HYPERPARAMS['EPI_START']:
                        steps_count+=1

                    # step counter for each episode
                    steps_used+=1.0

                    if (np.mod(steps_count,self.HYPERPARAMS['UPDATE_FREQ'])==0 and steps_count>0):
                        #update the layers by running the update ops...
                        sess.run(graph_vars['update_target'])

                    # stop playing this game, if the move just performed was terminal
                    if (done):
                        break
                # --------------------------------------------------------------

                # this episode has now been played

                # make updates to quantities

                steps_list.append(steps_used)
                reward_list.append(tot_reward)


                # if this game is a muliple of OUTPUT_STEP then average useful
                # quantities over the last OUTPUT_STEP games and write to output
                # arrays.
                if (np.mod(epi+1,self.PARAMS['OUTPUT_STEP'])==0 and epi>0):
                    steps_p_ep[out_count]  = np.sum(np.asarray(steps_list))/(1.0*(0.00001+len(steps_list)))
                    reward_p_ep[out_count] = np.sum(np.asarray(reward_list))/(1.0*(0.00001+len(reward_list)))
                    max_Q_p_ep[out_count]  = np.sum(np.asarray(maxQs))/(1.0*(0.00001+len(maxQs)))
                    min_Q_p_ep[out_count]  = np.sum(np.asarray(minQs))/(1.0*(0.00001+len(minQs)))
                    loss_p_ep[out_count]   = sum(losses)/(1.0*(0.01+len(losses)))
                    av_action_p_ep[out_count] =  sum(av_acts)/(1.0*(0.0001+len(av_acts)))
                    epsilon_ep[out_count]     = eps_tmp

                    summarynew = tf.Summary(value=[tf.Summary.Value(tag='avg steps', simple_value=steps_p_ep[out_count])])
                    summarynew.value.add(tag='avg reward', simple_value=reward_p_ep[out_count])
                    summarynew.value.add(tag='avg max Q', simple_value=max_Q_p_ep[out_count])
                    summarynew.value.add(tag='avg min Q', simple_value=min_Q_p_ep[out_count])
                    summarynew.value.add(tag='avg loss', simple_value=loss_p_ep[out_count])
                    summarynew.value.add(tag='avg action',simple_value=av_action_p_ep[out_count])
                    summarynew.value.add(tag='epsilon', simple_value=eps_tmp)

                    self.summary_hist(summarynew,'score hist',reward_list,60)
                    self.summary_hist(summarynew,'steps hist',steps_list,100)

                    # ALSO: at the output points, make a validation check
                    # run a game with no random moves: what is score
                    avg_valid_reward = 0.0
                    N_valid = 3
                    for j in np.arange(N_valid):
                        valid_reward = 0.0
                        current_obs = self.env.reset()
                        current_obs = self.preprocess(current_obs)
                        current_phi = np.tile( current_obs[:,:,np.newaxis], (1,1,self.PARAMS['Nc']) )
                        for i in np.arange(self.PARAMS['MAX_STEPS']):
                            # get action using the Q net

                            tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:]/255.0,
                                            graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:]/255.0,
                                            graph_vars['a_i_']:memory_normal.memory_a_i[:1,:],
                                            graph_vars['r_i_']:memory_normal.memory_r_i[:1,:],
                                            graph_vars['t_i_']:memory_normal.memory_terminal_i[:1,:]}

                            # use Q network graph to get Q_i, uses the online network
                            Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))
                            # the action to be taken, is one that maximises Q
                            action = np.argmax(Q)
                            new_obs, reward, done, info = self.env.step(self.action2step(action))

                            new_obs = self.preprocess(new_obs)
                            new_phi = np.concatenate((current_phi[:,:,1:],new_obs[:,:,np.newaxis]), axis=2)
                            current_phi = 1.0*new_phi

                            valid_reward+=reward
                            if (done):
                                break

                        avg_valid_reward+=valid_reward/float(N_valid)


                    #avg_valid_reward = avg_valid_reward*1.0/float(N_valid)
                    summarynew.value.add(tag='avg validation reward', simple_value=avg_valid_reward)


                    #print("wirting summary")
                    writer.add_summary(summarynew, epi+1)
                    # writer.flush()
                    if epi<self.PARAMS['OUTPUT_STEP']:
                        print("getting meta data for loss (not update or train)")
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(graph_vars['loss_'],feed_dict=tmp_feed_dict,options=run_options,run_metadata=run_metadata)
                        writer.add_run_metadata(run_metadata, 'epi-%d'%epi)
                    # else:
                    #     writer.add_summary(summarynew, epi+1)
                    writer.flush()

                    time_ep2 = time.time()
                    print("epsiode {a:d} --- avg/max steps = {b:.1f} / {maxsteps:.1f} --- avg/max/valid reward = {c:.1f} / {maxre:.1f} / {validre:.1f} --- epsilon = {d:.2f} --- time  = {e:.2f} \n".format(a=epi+1,b=steps_p_ep[out_count],maxsteps=np.amax(np.asarray(steps_list)),c=reward_p_ep[out_count],maxre=np.amax(np.asarray(reward_list)),validre=avg_valid_reward,d=eps_tmp,e=time_ep2-time_ep1))
                    time_ep1 = time.time()
                    out_count+=1



            #-------------------------------------------------------------------

            # n.b we are still inside with session as sess statement

            # training has finished - save a checkpoint to load later if want
            # to use the learned weights to actually play the game
            saved_path  = graph_vars['saver'].save(sess,"./../ckpts"+"/"+self.params_text)



        out_dict = {'rewards':reward_p_ep,'steps':steps_p_ep,'maxQ':max_Q_p_ep,'minQ':min_Q_p_ep,'losses':loss_p_ep,'actions':av_action_p_ep,'epsilon':epsilon_ep}
        return out_dict

    #---------------------------------------------------------------------------

    def play_animated_game(self):
        """ Render a game using a checkpoint for policy

            The HYPERPARAMS passed to the class inititaion should be the same
            as the HYPERPARAMS that were used for training the model.

            using the env.render functionality a game will be played locally
            on screen.

        """
        save_loc    = "./../ckpts"+"/"+self.params_text #+".ckpt"

        graph_vars = self.make_graph()

        #saver = tf.train.Saver()
        with tf.Session(graph=self.graph) as sess:
            new_saver = tf.train.import_meta_graph(save_loc+'.meta')
            new_saver.restore(sess,tf.train.latest_checkpoint("./../ckpts/"))


            ims = []
            fig = plt.figure()

            current_obs = self.env.reset()
            current_obs = self.preprocess(current_obs)
            current_phi = np.tile( current_obs[:,:,np.newaxis], (1,1,self.PARAMS['Nc']) )
            valid_steps = 0

            for i in np.arange(self.PARAMS['MAX_STEPS']):
                tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:]/255.0,
                                graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:]/255.0,
                                graph_vars['a_i_']:np.zeros((1,1)),
                                graph_vars['r_i_']:np.zeros((1,1)),
                                graph_vars['t_i_']:np.zeros((1,1))}

                # use Q network graph to get Q_i, uses the online network
                Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))
                # the action to be taken, is one that maximises Q
                action = np.argmax(Q)
                new_obs, reward, done, info = self.env.step(self.action2step(action))

                new_obs = self.preprocess(new_obs)
                new_phi = np.concatenate((current_phi[:,:,1:],new_obs[:,:,np.newaxis]), axis=2)
                current_phi = 1.0*new_phi

                time.sleep(0.04)


                self.env.render()
                if (done):
                    break
            self.env.close()


        return None

    def save_animated_game_mp4(self,dir='..'):
        """save a game to mp4 format

        The HYPERPARAMS passed to the class inititaion should be the same
        as the HYPERPARAMS that were used for training the model, in order to
        view how that model plays.

        Using the game will be stored as mp4 using matplotlib animation, via ffmpeg.

        The mp4 will be saved in a directory 'figs', on the same level as the 'code'
        directory.

        Args:
            dir: (optional) directory in which to look for a directory 'ckpts'
                where ckpt will be attempted to be loaded from, defaults to '..'

        """
        save_loc    = "./"+dir+"/ckpts"+"/"+self.params_text #+".ckpt"

        graph_vars = self.make_graph()

        #saver = tf.train.Saver()
        with tf.Session(graph=self.graph) as sess:
            new_saver = tf.train.import_meta_graph(save_loc+'.meta')
            #new_saver.restore(sess, save_loc+'.data-00000-of-00001')
            new_saver.restore(sess,tf.train.latest_checkpoint("./../ckpts/"))
            #saver.restore(sess,save_loc)

            ims = []
            fig = plt.figure()

            current_obs = self.env.reset()
            current_obs = self.preprocess(current_obs)
            current_phi = np.tile( current_obs[:,:,np.newaxis], (1,1,self.PARAMS['Nc']) )
            valid_steps = 0

            for i in np.arange(self.PARAMS['MAX_STEPS']):
                tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:]/255.0,
                                graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:]/255.0,
                                graph_vars['a_i_']:np.zeros((1,1)),
                                graph_vars['r_i_']:np.zeros((1,1)),
                                graph_vars['t_i_']:np.zeros((1,1))}

                # use Q network graph to get Q_i, uses the online network
                Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))
                # the action to be taken, is one that maximises Q
                action = np.argmax(Q)
                new_obs, reward, done, info = self.env.step(self.action2step(action))

                new_obs = self.preprocess(new_obs)
                new_phi = np.concatenate((current_phi[:,:,1:],new_obs[:,:,np.newaxis]), axis=2)
                current_phi = 1.0*new_phi


                im = plt.imshow(new_obs, animated=True)
                ims.append([im])

                if (done):
                    break
            self.env.close()
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

            ani.save('./../figs/'+self.params_text+'.mp4',writer=writer)

        return None

    def save_game_array(self,dir='..'):
        """Save a game as a 3d array of pixels

        The HYPERPARAMS passed to the class inititaion should be the same
        as the HYPERPARAMS that were used for training the model, in order to
        view how that model plays.

        the frames of the game played by the policy is stored in an array, of
        dimension (x,y,number of frames). The x and y dimensions are downsampled
        from the raw frame state to reduce the output size. array will be saved
        by numpy.save(), into a directory 'game_arrays' at the same level as 'code'
        which can be loaded loaded seperately in 'mp4_from_array' method to create
        a video of the game.

        Note that when running on an aws instance it is convenient to save the
        array to file on the aws machine, and scp the saved array back to the
        local machine, where one can then create the video.

        Args:
            dir: (optional) directory in which to look for a directory 'ckpts'
                where ckpt will be attempted to be loaded from, defaults to '..'

        """
        save_loc    = "./"+dir+"/ckpts"+"/"+self.params_text #+".ckpt"

        graph_vars = self.make_graph()

        #saver = tf.train.Saver()
        with tf.Session(graph=self.graph) as sess:
            new_saver = tf.train.import_meta_graph(save_loc+'.meta')
            #new_saver.restore(sess, save_loc+'.data-00000-of-00001')
            new_saver.restore(sess,tf.train.latest_checkpoint("./../ckpts/"))
            #saver.restore(sess,save_loc)

            ims = []
            # fig = plt.figure()

            current_obs = self.env.reset()
            current_obs = self.preprocess(current_obs)
            current_phi = np.tile( current_obs[:,:,np.newaxis], (1,1,self.PARAMS['Nc']) )
            valid_steps = 0
            tot_reward = 0

            for i in np.arange(self.PARAMS['MAX_STEPS']):
                tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:]/255.0,
                                graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:]/255.0,
                                graph_vars['a_i_']:np.zeros((1,1)),
                                graph_vars['r_i_']:np.zeros((1,1)),
                                graph_vars['t_i_']:np.zeros((1,1))}

                # use Q network graph to get Q_i, uses the online network
                Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))
                # the action to be taken, is one that maximises Q
                action = np.argmax(Q)
                new_obs, reward, done, info = self.env.step(self.action2step(action))

                new_obs2 = self.preprocess(new_obs)
                new_phi = np.concatenate((current_phi[:,:,1:],new_obs2[:,:,np.newaxis]), axis=2)
                current_phi = 1.0*new_phi

                tot_reward+= reward
                # plt.imshow(new_obs[::2,::2,:])
                # plt.show()

                # im = plt.imshow(new_obs, animated=True)
                if np.mod(i,2)==0:
                    ims.append(new_obs[::2,::2,:])

                if (done):
                    break
            self.env.close()

            print("shape = ", np.shape(np.asarray(ims)))
            print(np.asarray(ims).dtype)

            print("game_score = ",tot_reward)

            #note array saved in format: frame number,x,y,color
            if os.path.isdir('./../game_arrays/'):
                np.save('./../game_arrays/'+self.params_text,np.asarray(ims))
            else:
                os.mkdir('./../game_arrays/')
                np.save('./../game_arrays/'+self.params_text,np.asarray(ims))


        return None

    def mp4_from_array(self,dir='..'):
        """Save a game to mp4 format, from a saved numpy array

        The HYPERPARAMS passed to the class inititaion should be the same
        as the HYPERPARAMS that were used for training the model, in order to
        view how that model plays.

        Args:
            dir: (optional) directory in which to look for a directory 'game_arrays'
                where an array will be attempted to be loaded from, defaults to '..'


        """
        save_loc    = "./"+dir+"/game_arrays"+"/"+self.params_text #+".ckpt"
        game_arr    = np.load(save_loc+'.npy')

        print(save_loc)
        #saver = tf.train.Saver()
        with tf.Session(graph=self.graph) as sess:

            ims = []
            fig = plt.figure()
            ax = plt.subplot(111)
            plt.axis('off')
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])

            print(np.shape(game_arr))
            print(len(game_arr))
            for i in np.arange(len(game_arr)):


                im = ax.imshow(game_arr[i,:,:,:], animated=True)
                ims.append([im])

            self.env.close()
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

            ani.save('./../figs/'+self.params_text+'.mp4',writer=writer)

        return None
