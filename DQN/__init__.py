import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import time

import matplotlib #.pyplot as plt
matplotlib.use('TkAgg') # this makes the fgire in focus rather than temrinal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gym

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Qmemory:
    """ memory object for Q learning

    This class provides a way to define a memory object for depp Q learning,
    to provide memory recall. memory is made up of memory of transitions during
    'play'. This is made of memory of states i, j (=i+1), the action take at i (to get to j)
    the reward earned by making that action, and whether the state j was the end of the game.

    transitions can be added to the memory with the 'write' method. This will automatically
    add the transition data to the memory, and once the memory is full, will start
    to rewrite the memory of events from many moves ago.

    One can also retrieve a subset of the memory using 'get_batch'. This selects a
    random subset, taking into account whether the memory is full, and if not only pulling
    from examples that are already 'filled in' in the memory.
    """
    def __init__(self,N_mem, N_x, N_y, N_frames):
        """create a memory object

        Args:
            N_mem: how large the memory should be, how many transitions to store
            obs_data_size: how large is each observation (state), as a single integer.
                        e.g a 2d frame of 80x80 80x80=1600
            N_frames: how many frames are stored per state
        """
        self.N_mem         = N_mem
        #self.obs_data_size = obs_data_size
        self.N_frames      = N_frames

        # initialize the memory arrays.
        self.memory_phi_i      = np.zeros((N_mem,N_x,N_y,N_frames),dtype=np.uint8)
        self.memory_phi_j      = np.zeros((N_mem,N_x,N_y,N_frames),dtype=np.uint8)
        self.memory_a_i        = np.zeros((N_mem,1),dtype=np.uint8)
        self.memory_r_i        = np.zeros((N_mem,1),dtype=np.float32)
        self.memory_terminal_i = np.zeros((N_mem,1),dtype=np.float32)

        self.mem_count = 0

    def write(self,phi_i,phi_j,action,reward,terminal_float):
        """write a set of transition data to the memory

        Args:
            phi_i: state i
            phi_j: state_j
            action: action taken at i
            reward: reward recieved
            terminal_float: is the move terminal, is it the last move in game, as a float: 0 means is last move, 1 means isnt.

        writes the data into position np.mod(self.mem_count,self.N_mem) in the memory.
        This means that it will loop back to position 0 once the memory is full, and
        memory will be rewritten.
        """
        self.memory_phi_i[np.mod(self.mem_count,self.N_mem),:,:,:] = phi_i
        self.memory_phi_j[np.mod(self.mem_count,self.N_mem),:,:,:] = phi_j
        self.memory_a_i[np.mod(self.mem_count,self.N_mem),:] = action.astype(np.uint8)
        self.memory_r_i[np.mod(self.mem_count,self.N_mem),:] = reward
        self.memory_terminal_i[np.mod(self.mem_count,self.N_mem),:] = terminal_float
        # we just added some data, updtae our counter to tell us how manyh we added in total
        self.mem_count+=1

    def get_batch(self,N_get):
        """get a subset of the memory for training Q network.

        Args:
            N_get: how many transition event to get (i.e. return)

        Returns:
            batch_dict: a dictionary containing memory arrays:
                        - phi_i: state i
                        - phi_j: state j (=i+1)
                        - r_i: reward
                        - t_i: whether terminal
        """
        # check if memory is full or not
        if self.mem_count>=self.N_mem:
            # is full
            max_val = self.N_mem
        else:
            # isn't full - max index to look up to is the current count
            max_val = self.mem_count

        # get random integeres between 0 and our max_val defined above
        rand_ints = np.random.randint(0,high=max_val,size=N_get)

        # use rand_ints to get random memory selection
        batch_dict = {'phi_i': self.memory_phi_i[rand_ints,:,:,:],
                      'phi_j': self.memory_phi_j[rand_ints,:,:,:],
                      'a_i': self.memory_a_i[rand_ints,:],
                      'r_i': self.memory_r_i[rand_ints,:],
                      't_i': self.memory_terminal_i[rand_ints,:]}
        return batch_dict

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class deepQ:
    """Object for deep Q learning, for solving openai gym environments

    deep Q network can be used for Q learning, to find the Q function that maximses
    the reward, and effectively therefore gives an optimal stragey for the game.

    The method used here contains various elements of the deepQ algorithm, namely:
     - experience replay
     - double Q learning
     - online and target networks with strided update


    object contains methods for preprocessing frames of gym games

    """

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def __init__(self,game,HYPERPARAMS,PARAMS):
        """initialize useful stuff"""

        self.env = gym.make(game)

        self.HYPERPARAMS = HYPERPARAMS
        self.PARAMS   = PARAMS

        self.N_action = self.env.action_space.n

        tf.reset_default_graph()
        self.graph = tf.get_default_graph() #tf.Graph()

        alpha_txt = f"alpha_{HYPERPARAMS['ALPHA']:.2e}_"
        upd_txt   = f"updfreq_{HYPERPARAMS['UPDATE_FREQ']:d}_"
        decay_txt = f"EPSDECAY_{HYPERPARAMS['EPS_DECAY']:.1f}_"
        nfc_txt   = f"NFC_{HYPERPARAMS['N_FC']:d}_"
        nfilt_txt = f"Nfilter_{HYPERPARAMS['N_FILTER']:d}_"
        mem_txt   = f"mem_{HYPERPARAMS['N_memory']:d}_"
        batch_txt = f"batch_{HYPERPARAMS['N_batch']:d}"
        self.params_text = alpha_txt+upd_txt+decay_txt+nfc_txt+\
                           nfilt_txt+mem_txt+batch_txt
        #self.params_text = f"alpha_{HYPERPARAMS['ALPHA']:.2e}_updfreq_{HYPERPARAMS['UPDATE_FREQ']:d}_EPSDECAY_{HYPERPARAMS['EPS_DECAY']:.1f}_NFC_{HYPERPARAMS['N_FC']:d}_NFilter_{HYPERPARAMS['N_FILTER']:d}_mem_{HYPERPARAMS['N_memory']:d}_batch_{HYPERPARAMS['N_batch']:d}"

        print("\n==========================================================")
        print("\n\n filename for saving       : ",self.params_text)
        print(" action space has size     : ",self.N_action)
        print(" using tensorflow version  : ",tf.VERSION, "\n\n")
        print("==========================================================\n")
        return None

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------


    def preprocess(self,frame):
        """preprocess frame of game

        Args:
            frame: a frame of the game

        applies self.mean_obs and self.std_obvs to normalize the data onto distribution
        with std=1 and mean=0, to allow for better initialization of network etc.
        """
        frame_out = np.zeros((84,84),dtype=np.uint8)
        tmp = np.mean(frame, axis=2)
        tmp = tmp[28:-12, :]
        tmp = tmp[1:-1:2,::2]
        frame_out[:,2:-2] = tmp.astype(np.uint8)
        return frame_out

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    def Qnet(self,obs,call_type,trainme,reuseme):
        """neural network to get Q for given state

        Args:
            obs: (tensor) set of observations to predict Q for: size: batch,(x,y..),frames
                frames should be 4 to match deepmind paper.
                (x,y...) to be determined by game - 2d frame => x,y
                but Cartpole gives 1d array as frame so is just x
            call_type: 'online_' or 'target_' - which network to use
            trainme: (bool) should the weights be trainable
            reuseme: (bool) should the weights be reusable

        """

        z = tf.reshape(obs, [-1,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc']])
        #print(z.shape)
        with tf.variable_scope(call_type+'conv_layer0',reuse=reuseme):
            z_conv0 = tf.layers.Conv2D(filters = self.HYPERPARAMS['N_FILTER'],
                                        kernel_size = (8,8),
                                        strides = (4,4),
                                        padding='valid',
                                        activation=tf.nn.leaky_relu,
                                        trainable=trainme,
                                        kernel_initializer=tf.keras.initializers.he_normal())(z)

        with tf.variable_scope(call_type+'conv_layer1',reuse=reuseme):
            z_conv1 = tf.layers.Conv2D(filters = 2*self.HYPERPARAMS['N_FILTER'],
                                        kernel_size = (4,4),
                                        strides = (2,2),
                                        padding='valid',
                                        activation=tf.nn.leaky_relu,
                                        trainable=trainme,
                                        kernel_initializer=tf.keras.initializers.he_normal())(z_conv0)
            z_conv1_flat = tf.reshape(z_conv1,[-1,self.PARAMS['N_squash']*self.PARAMS['N_squash']*(2*self.HYPERPARAMS['N_FILTER'])])

        with tf.variable_scope(call_type+'FC_layer0',reuse=reuseme):
            z_FC0 =  tf.layers.Dense(units=self.HYPERPARAMS['N_FC'],trainable=trainme,kernel_initializer=tf.keras.initializers.he_normal())(z_conv1_flat)

        with tf.variable_scope(call_type+'layer_out',reuse=reuseme):
            z_out = tf.layers.Dense(units=self.N_action,trainable=trainme,kernel_initializer=tf.keras.initializers.he_normal())(z_FC0)

        return z_out

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    def update_layer(self,layer):
        """update the wieghts/biases of target network

        Args:
            layer: (string) name of layer. e.g. 'layer0'

        Returns:
            upd_k: operator that updates the kernel of the layer
            epd_b: operator that updates the bias of the layer
        """

        # TODO: tf.group to combine operators...????
        with tf.variable_scope('online_' + layer,reuse=True):
            k_online = tf.get_variable('kernel')
            b_online = tf.get_variable('bias')

        with tf.variable_scope('target_' + layer,reuse=True):
            k_target = tf.get_variable('kernel')
            b_target = tf.get_variable('bias')

        upd_k = tf.assign(k_target,k_online)
        upd_b = tf.assign(b_target,b_online)
        return upd_k,upd_b

    #---------------------------------------------------------------------------

    def make_graph(self):
        """ Define the computational graph

        takes in the game states (before and after action), action, reward, and
        whether terminal as placeholders. Uses these to compute Q values for
        both online and target networks. Applies the double deep Q learning
        algorithm, using self.Qnet as the neural network which predicts the
        Q values for a given state.

        Args:
            Placeholders:
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
            Q_i_ = self.Qnet(phi_i_,'online_',True,False)
            print("Q_i_ shape         = ",Q_i_.shape)

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

            Qj_online_ = self.Qnet(phi_j_,'online_',True,True)
            Qj_online_inds = tf.argmax(Qj_online_,axis=1)
            Qj_onehot_inds = tf.one_hot(Qj_online_inds, self.N_action)

            # ------------------------------------------------------------------

            # this has reuse=False, make a new network - the target network
            # it is not trainable. Instead we train the online network and
            # set the weights/biases of the layers in the target network to be the
            # same as those in the online network every so many games.

            Q_j_ = self.Qnet(phi_j_,'target_',False,False)

            # now only take values of Q (target) for state j, using action that
            # the online network would predict
            V_j_ = tf.reduce_sum(tf.multiply(Qj_onehot_inds,Q_j_),axis=1)

            # ------------------------------------------------------------------
            # get the future discounted reward
            y_          = tf.add( tf.squeeze(r_i_) , self.HYPERPARAMS['GAMMA']*tf.multiply(tf.squeeze(t_i_),tf.squeeze(V_j_)))

            print("y shape = ",y_.shape)
            print("r_i_ shape = ",tf.squeeze(r_i_).shape)

            # difference between value function (future discounted) and the value
            # funtion on state i
            x_    = tf.subtract( y_, V_i_  )

            print("x_ shape = ",x_.shape)

            # ------------------------------------------------------------------
            # define the loss, create an optimizer op, and a training op

            # use a Pseudo-Huber loss
            loss_scale = self.HYPERPARAMS['LOSS_SCALE'] # how steep loss is for large values
            loss_ = tf.reduce_mean( loss_scale*(tf.sqrt(1.0+(1.0/loss_scale)**2*tf.multiply(x_,x_)) - 1.0) )

            optimizer    = tf.train.RMSPropOptimizer(self.HYPERPARAMS['ALPHA'])

            train_op     = optimizer.minimize(loss_)

            # ------------------------------------------------------------------
            # update the parameters of the target network, by cloning those from
            # online Q network. This will only be sess.run'ed every C steps

            upd_c_k0,upd_c_b0   = self.update_layer('conv_layer0/conv2d')
            upd_c_k1,upd_c_b1   = self.update_layer('conv_layer1/conv2d')
            upd_FC_k0,upd_FC_b0 = self.update_layer('FC_layer0/dense')
            upd_k_out,upd_b_out = self.update_layer('layer_out/dense')

            update_target = tf.group(upd_c_k0, upd_c_b0, upd_c_k1, upd_c_b1, upd_FC_k0, upd_FC_b0, upd_k_out, upd_b_out)

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




    #---------------------------------------------------------------------------



    def train(self, N_episodes):
        """train the DeepQ network

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
            #K.set_session(sess)
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

            # the idea of this is that as game play gets longer as mmodel improves,
            # the memory would then contain fewer losing moves to learn from. Which
            # could cause the model to 'forget' how to play.

            N_mem_normal = int(0.90*self.HYPERPARAMS['N_memory'])
            N_mem_losses = self.HYPERPARAMS['N_memory'] - N_mem_normal #int(0.05*self.HYPERPARAMS['N_memory'])

            memory_normal = Qmemory(N_mem_normal,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc'])
            memory_losses = Qmemory(N_mem_losses,self.PARAMS['N_x'],self.PARAMS['N_y'],self.PARAMS['Nc'])

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
                    eps_tmp = 1.0
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
                        action = np.asarray(self.env.action_space.sample())
                        new_obs, reward, done, info = self.env.step(action)

                        av_acts.append(action)
                    else:
                        # feed data into the session graph to get Q as a numpy array
                        # only phi_i is actual data
                        # phi_j, a_i, r_i are just dummy really as not used

                        tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:],
                                        graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:],
                                        graph_vars['a_i_']:memory_normal.memory_a_i[:1,:],
                                        graph_vars['r_i_']:memory_normal.memory_r_i[:1,:],
                                        graph_vars['t_i_']:memory_normal.memory_terminal_i[:1,:]}

                        # use Q network graph to get Q_i, uses the online network
                        Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))
                        #Qhat = sess.run([Q_j_],tmp_feed_dict)

                        # append the max and min Q to the lists (will be averaged later)
                        maxQs.append(np.amax(Q))
                        minQs.append(np.amin(Q))

                        # the action to be taken, is one that maximises Q
                        action = np.argmax(Q)
                        #print(action)
                        new_obs, reward, done, info = self.env.step(action)
                        av_acts.append(action)


                    # ----------------------------------------------------------

                    # process some of the outputs to make them behave nicely
                    # with the network and computational graph

                    # preprocess the image
                    new_obs = self.preprocess(new_obs)

                    # phi is made of several observations/frames. so concatenate
                    # the the current phi (all but first frame), with the new observation
                    # this then becomes the new state containg 'Nc' frames
                    new_phi = np.concatenate((current_phi[:,:,1:],new_obs[:,:,np.newaxis]), axis=2)

                    # adapt the reward on losing move to be more negative to
                    # penalize failing more
                    # if done:
                    #     reward=-5.0


                    # convert the boolean 'done' which tells us if move is the last
                    # of game, to a float (number). we want it to be 0.0 when it is
                    # the final move - so we need the opposite of normal conversion
                    # of bool->float - so use:   (not done)
                    term_float = np.array(not done)
                    term_float = term_float.astype(np.float32)

                    tot_reward+=reward

                    # ----------------------------------------------------------
                    # WRITE new experience to MEMORY

                    # only perform if we are more than 'Nc' moves into this game.
                    # this is because we need to have phi states of length 'Nc'
                    # and they are initialised at beginning of game to be the initial
                    # frame repeated Nc times, which would be unrealistic - so do
                    # not add to memory.
                    if i>=(self.PARAMS['Nc']-1):
                        if reward > 0.0:
                            memory_normal.write(current_phi, new_phi, action, reward, term_float)
                        else:
                            memory_losses.write(current_phi, new_phi, action, reward, term_float)

                    # ----------------------------------------------------------

                    # APPLY LEARING UPDATES

                    # take a batch of the experiences from memory
                    # only do if experience memory is big enough to contain N_batch entries
                    #if (mem_count>self.HYPERPARAMS['N_batch']):
                    if(epi>self.HYPERPARAMS['EPI_START']):

                        # define sizes of batches for the 'normal' memory, and
                        # batch size for the 'losses' memory.

                        N_batch_n = int(self.HYPERPARAMS['N_batch']*0.8)
                        N_batch_l = self.HYPERPARAMS['N_batch'] - N_batch_n

                        batch_n = memory_normal.get_batch(N_batch_n)
                        batch_l = memory_losses.get_batch(N_batch_l)

                        # combine the batches from both memories to create a single
                        # batch which represents 'normal' and 'loss' moves with a
                        # predetermined ratio.

                        phi_i_batch = np.concatenate((batch_n['phi_i'], batch_l['phi_i'])  , axis=0)/255.0
                        phi_j_batch = np.concatenate((batch_n['phi_j'], batch_l['phi_j'] ) , axis=0)/255.0
                        a_i_batch   = np.concatenate((batch_n['a_i']  , batch_l['a_i'])    , axis=0)
                        r_i_batch   = np.concatenate((batch_n['r_i']  , batch_l['r_i'])    , axis=0)
                        t_i_batch   = np.concatenate((batch_n['t_i']  , batch_l['t_i'])    , axis=0)

                        feed_dict_batch = { graph_vars['phi_i_']:(phi_i_batch).astype(np.float32),
                                            graph_vars['phi_j_']:(phi_j_batch).astype(np.float32),
                                            graph_vars['r_i_']:r_i_batch,
                                            graph_vars['a_i_']:a_i_batch,
                                            graph_vars['t_i_']:t_i_batch}

                        # get the loss for this batch
                        loss0 = sess.run(graph_vars['loss_'],feed_dict=feed_dict_batch)
                        # append loss to be averaged later
                        losses.append(loss0)

                        # get the loss sent as a scalr to tensorboard summary
                        # tmp_summary = sess.run(graph_vars['merged'],feed_dict=feed_dict_batch)
                        # writer.add_summary(tmp_summary, epi)




                        # APPLY GRADIENT DESCENT for batch
                        # only perform if episopde is > EPI_START
                        if(epi>self.HYPERPARAMS['EPI_START']):
                            graph_vars['train_op'].run(feed_dict=feed_dict_batch)


                    # ----------------------------------------------------------

                    # prepare for beginning a new game, update counters etc

                    # RESET what the current phi is for the next step
                    current_phi = 1.0*new_phi

                    # if we are in the training period - add one to total number
                    # of steps taken
                    if epi>self.HYPERPARAMS['EPI_START']:
                        steps_count+=1

                    steps_used+=1.0

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
                    summarynew.value.add(tag='avg loss', simple_value=loss_p_ep[out_count])
                    summarynew.value.add(tag='epsilon', simple_value=eps_tmp)


                    # ALSO: at the output points, make a validation check
                    # run a game with no random moves: what is score
                    avg_valid_reward = 0
                    N_valid = 10
                    for j in np.arange(N_valid):
                        valid_reward = 0.0
                        current_obs = self.env.reset()
                        current_obs = self.preprocess(current_obs)
                        current_phi = np.tile( current_obs[:,:,np.newaxis], (1,1,self.PARAMS['Nc']) )
                        for i in np.arange(self.PARAMS['MAX_STEPS']):
                            # get action using the Q net

                            tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:],
                                            graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:],
                                            graph_vars['a_i_']:memory_normal.memory_a_i[:1,:],
                                            graph_vars['r_i_']:memory_normal.memory_r_i[:1,:],
                                            graph_vars['t_i_']:memory_normal.memory_terminal_i[:1,:]}

                            # use Q network graph to get Q_i, uses the online network
                            Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))
                            # the action to be taken, is one that maximises Q
                            action = np.argmax(Q)
                            new_obs, reward, done, info = self.env.step(action)

                            new_obs = self.preprocess(new_obs)
                            new_phi = np.concatenate((current_phi[:,:,1:],new_obs[:,:,np.newaxis]), axis=2)
                            current_phi = 1.0*new_phi

                            valid_reward+=reward
                            if (done):
                                break
                        avg_valid_reward+=valid_reward

                    avg_valid_reward = avg_valid_reward*1.0/float(N_valid)
                    summarynew.value.add(tag='avg validation reward', simple_value=avg_valid_reward)

                    writer.add_summary(summarynew, epi+1)

                    time_ep2 = time.time()
                    print(" on epsiode {a:d} ----- avg steps = {b:.1f} ------ avg reward = {c:.1f}  ---- epsilon = {d:.2f} ----- time  = {e:.2f} \n".format(a=epi+1,b=steps_p_ep[out_count],c=reward_p_ep[out_count],d=eps_tmp,e=time_ep2-time_ep1))
                    time_ep1 = time.time()
                    out_count+=1




                # if epsioside is a multiple of UPDATE_FREQ update the weights/biases
                # of the target network to be those of the online network
                if (np.mod(epi,self.HYPERPARAMS['UPDATE_FREQ'])==0):
                    #update the layers by running the update ops...
                    sess.run(graph_vars['update_target'])
                    # sess.run([upd_c_k0,upd_c_b0,
                    #           upd_c_k1,upd_c_b1,
                    #           upd_FC_k0, upd_FC_b0,
                    #           upd_k_out, upd_b_out])

            #-------------------------------------------------------------------

            # n.b we are still inside with session as sess statement

            # training has finished - save a checkpoint to load later if want
            # to use the learned weights to actually play the game
            saved_path  = graph_vars['saver'].save(sess,"./../ckpts"+"/"+self.params_text)



        out_dict = {'rewards':reward_p_ep,'steps':steps_p_ep,'maxQ':max_Q_p_ep,'minQ':min_Q_p_ep,'losses':loss_p_ep,'actions':av_action_p_ep,'epsilon':epsilon_ep}
        return out_dict

    #---------------------------------------------------------------------------

    def play_animated_game(self):
        #params_text = f"NFC_{self.HYPERPARAMS['N_FC']:d}"
        save_loc    = "./../ckpts"+"/"+self.params_text #+".ckpt"

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
                tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:],
                                graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:],
                                graph_vars['a_i_']:np.zeros((1,1)),
                                graph_vars['r_i_']:np.zeros((1,1)),
                                graph_vars['t_i_']:np.zeros((1,1))}

                # use Q network graph to get Q_i, uses the online network
                Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))
                # the action to be taken, is one that maximises Q
                action = np.argmax(Q)
                new_obs, reward, done, info = self.env.step(action)

                new_obs = self.preprocess(new_obs)
                new_phi = np.concatenate((current_phi[:,:,1:],new_obs[:,:,np.newaxis]), axis=2)
                current_phi = 1.0*new_phi

                time.sleep(0.04)

                # im = plt.imshow(frame, animated=True)
                # ims.append([im])
                self.env.render()
                if (done):
                    break
            self.env.close()
            # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
            #
            # Writer = animation.writers['ffmpeg']
            # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            #
            # ani.save('./../figs/'+params_text+'.mp4',writer=writer)

        return None

    def save_animated_game(self,dir='..'):
        """save a game to mp4 format

        this function loads game from checkpoint, so make sure you already ran
        a game with the hyperparams you want to check.

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
                tmp_feed_dict = {graph_vars['phi_i_']:current_phi[np.newaxis,:,:,:],
                                graph_vars['phi_j_']:current_phi[np.newaxis,:,:,:],
                                graph_vars['a_i_']:np.zeros((1,1)),
                                graph_vars['r_i_']:np.zeros((1,1)),
                                graph_vars['t_i_']:np.zeros((1,1))}

                # use Q network graph to get Q_i, uses the online network
                Q = np.squeeze(sess.run([graph_vars['Q_i_']],tmp_feed_dict))
                # the action to be taken, is one that maximises Q
                action = np.argmax(Q)
                new_obs, reward, done, info = self.env.step(action)

                # new_obs = self.preprocess(new_obs)
                # new_phi = np.concatenate((current_phi[:,:,1:],new_obs[:,:,np.newaxis]), axis=2)
                # current_phi = 1.0*new_phi


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
