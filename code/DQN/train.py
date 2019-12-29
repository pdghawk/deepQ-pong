
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class TrainingGraph:
    def __init__(self,graph_vars):
        self.graph_vars=graph_vars
        self._check_expected()

    def _check_expected(self):
        expected=[
        'graph_init',
        'graph_local_init',
        'Q_i',
        'loss',
        'train_op',
        'update_target',
        'phi_i',
        'phi_j',
        'a_i',
        'r_i',
        't_i',
        'saver'
        ]
        for e in expected:
            if e not in self.graph_vars:
                raise ValueError(
                "make sure all appropriate fields are filled"+
                "in input to TrainingGraph, missing:"+
                str(set(expected).intersection(set(self.graph_vars.keys()))))


    def initialize_graph(self,sess):
        sess.run(self.graph_vars['graph_init'])
        sess.run(self.graph_vars['graph_local_init'])

    def construct_feed_dict(self,phi_i,phi_j,a_i,r_i,t_i):
        feed_dict = {self.graph_vars['phi_i']:phi_i,
                    self.graph_vars['phi_j']:phi_j,
                    self.graph_vars['a_i']:a_i,
                    self.graph_vars['r_i']:r_i,
                    self.graph_vars['t_i']:t_i}
        return feed_dict

    def get_q(self,sess,phi_i,phi_j,a_i,r_i,t_i):
        fd = self.construct_feed_dict(phi_i,phi_j,a_i,r_i,t_i)
        q = np.squeeze(sess.run([self.graph_vars['Q_i']],fd))
        return q

    def get_loss(self,sess,phi_i,phi_j,a_i,r_i,t_i,options=None,meta=None):
        fd = self.construct_feed_dict(phi_i,phi_j,a_i,r_i,t_i)
        if(options is not None and meta is not None):
            loss = np.squeeze(sess.run([self.graph_vars['loss']],fd,
                              options=options,meta=meta))
        elif(options is not None):
            loss = np.squeeze(sess.run([self.graph_vars['loss']],fd,
                              options=options))
        elif(meta is not None):
            loss = np.squeeze(sess.run([self.graph_vars['loss']],fd,meta=meta))
        else:
            loss = np.squeeze(sess.run([self.graph_vars['loss']],fd))
        return loss

    def do_train_step(self,sess,phi_i,phi_j,a_i,r_i,t_i):
        fd = self.construct_feed_dict(phi_i,phi_j,a_i,r_i,t_i)
        self.graph_vars['train_op'].run(feed_dict=fd)

    def update_target(self,sess):
        sess.run(self.graph_vars['update_target'])


class TrainingGraphFactory(ABC):
    def __init__(self,q_net,learning_rate,discount_factor):
        self.q_net = q_net
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # self.N_x=None
        # self.N_y=None
        # self.frames=None

    # def setup(self,N_x,N_y,frames,output_dimension):
    #     self.N_x=N_x
    #     self.N_y=N_y
    #     self.frames=frames
    #     self.output_dimension = output_dimension

    @abstractmethod
    def _build_graph(self,Nx,Ny,frames,output_dimension):
        pass

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

    def make(self,graph,Nx,Ny,frames,output_dimension):
        # if( any(i is None for i in [self.N_x,self.N_y,self.frames] )):
        #     raise RuntimeError("call TrainingGraphFactory.setup()")
        with graph.as_default():
            return self._build_graph(Nx,Ny,frames,output_dimension)

class DDQNTrainingGraphFactory(TrainingGraphFactory):

    def _build_graph(self,Nx,Ny,frames,output_dimension):
        # placeholders for the states, actions, rewards, and whether terminal
        # size is batch, (x, y,), stored frames(4)
        phi_i_ = tf.placeholder(shape=[None,Nx,Ny,frames],dtype=tf.float32)
        phi_j_ = tf.placeholder(shape=[None,Nx,Ny,frames],dtype=tf.float32)
        a_i_   = tf.placeholder(shape=[None,1],dtype=tf.uint8)
        r_i_   = tf.placeholder(shape=[None,1],dtype=tf.float32)
        t_i_   = tf.placeholder(shape=[None,1],dtype=tf.float32)

        # ------------------------------------------------------------------
        with tf.name_scope('Q_i_online'):
            Q_i_ = self.q_net.run(phi_i_,'online',True,False)
            #print("Q_i_ shape         = ",Q_i_.shape)

        with tf.name_scope('Value_function_i_online'):
            # convert actions that were taken into onehot format
            a_list = tf.reshape(tf.cast(a_i_,tf.int32),[-1])
            print("a_list shape = ",a_list.shape)

            a_onehot = tf.one_hot(a_list, output_dimension)
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
            Qj_online_ = self.q_net.run(phi_j_,'online',True,True)
            Qj_online_inds = tf.argmax(Qj_online_,axis=1)
            Qj_onehot_inds = tf.one_hot(Qj_online_inds, output_dimension)

        # ------------------------------------------------------------------

        # this has reuse=False, make a new network - the target network
        # it is not trainable. Instead we train the online network and
        # set the weights/biases of the layers in the target network to be the
        # same as those in the online network every so many games.
        with tf.name_scope('Qj_target'):
            Q_j_ = self.q_net.run(phi_j_,'target',False,False)

        # now only take values of Q (target) for state j, using action that
        # the online network would predict
        with tf.name_scope('value_function_j'):
            V_j_ = tf.reduce_sum(tf.multiply(Qj_onehot_inds,Q_j_),axis=1)

        # ------------------------------------------------------------------
        # get the future discounted reward
        with tf.name_scope('discounted_reward'):
            y_          = tf.add( tf.squeeze(r_i_) , self.discount_factor*tf.multiply(tf.squeeze(t_i_),tf.squeeze(V_j_)))

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
            loss_scale = 2.0 #self.HYPERPARAMS['LOSS_SCALE'] # how steep loss is for large values
            loss_ = tf.reduce_mean( loss_scale*(tf.sqrt(1.0+(1.0/loss_scale)**2*tf.multiply(x_,x_)) - 1.0) )

        with tf.name_scope('optimizer'):
            optimizer    = tf.train.RMSPropOptimizer(self.learning_rate)
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
                        'Q_i':Q_i_,
                        'loss':loss_,
                        'train_op':train_op,
                        'update_target':update_target,
                        'merged':merged,
                        'phi_i':phi_i_,
                        'phi_j':phi_j_,
                        'a_i':a_i_,
                        'r_i':r_i_,
                        't_i':t_i_,
                        'saver':saver}

        output_graph = TrainingGraph(graph_vars)
        return output_graph
