
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class TrainingGraph:
    def __init__(self,graph_vars):
        self._graph_vars=graph_vars
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
            if e not in self._graph_vars:
                raise ValueError(
                "make sure all appropriate fields are filled"+
                "in input to TrainingGraph, missing:"+
                str(set(expected).intersection(set(self._graph_vars.keys()))))


    def initialize_graph(self,sess):
        sess.run(self._graph_vars['graph_init'])
        sess.run(self._graph_vars['graph_local_init'])

    def construct_feed_dict(self,phi_i,phi_j,a_i,r_i,t_i):
        feed_dict = {self._graph_vars['phi_i']:phi_i,
                    self._graph_vars['phi_j']:phi_j,
                    self._graph_vars['a_i']:a_i,
                    self._graph_vars['r_i']:r_i,
                    self._graph_vars['t_i']:t_i}
        return feed_dict

    def get_q(self,sess,phi_i,phi_j,a_i,r_i,t_i):
        fd = self.construct_feed_dict(phi_i,phi_j,a_i,r_i,t_i)
        q = np.squeeze(sess.run([self._graph_vars['Q_i']],fd))
        return q

    def get_loss(self,sess,phi_i,phi_j,a_i,r_i,t_i,options=None,meta=None):
        fd = self.construct_feed_dict(phi_i,phi_j,a_i,r_i,t_i)
        if(options is not None and meta is not None):
            loss = np.squeeze(sess.run([self._graph_vars['loss']],fd,
                              options=options,meta=meta))
        elif(options is not None):
            loss = np.squeeze(sess.run([self._graph_vars['loss']],fd,
                              options=options))
        elif(meta is not None):
            loss = np.squeeze(sess.run([self._graph_vars['loss']],fd,meta=meta))
        else:
            loss = np.squeeze(sess.run([self._graph_vars['loss']],fd))
        return loss

    def do_train_step(self,sess,phi_i,phi_j,a_i,r_i,t_i):
        fd = self.construct_feed_dict(phi_i,phi_j,a_i,r_i,t_i)
        self._graph_vars['train_op'].run(feed_dict=fd)

    def update_target(self,sess):
        sess.run(self._graph_vars['update_target'])


class TrainingGraphFactory(ABC):
    def __init__(self,q_net,learning_rate,discount_factor):
        self.q_net = q_net
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    @abstractmethod
    def value_j_online(self,*args):
        pass

    def get_layer_names(self):
        #return self.qnet.get_layer_names()
        return ['conv_layer0/conv2d','conv_layer1/conv2d','conv_layer2/conv2d',
                 'FC_layer0/dense','layer_out/dense']

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

    def update_all_layers(self):
        layer_names = self.get_layer_names()
        with tf.name_scope('target_updates'):
            upd_k,upd_b=self.update_layer(layer_names[0])
            update_target=tf.group(upd_k,upd_b)
            for name in layer_names[1:]:
                upd_k,upd_b   = self.update_layer(name)
                update_target = tf.group(update_target, upd_k, upd_b)
        return update_target

    def set_placeholders(self,Nx,Ny,frames):
        phi_i = tf.placeholder(shape=[None,Nx,Ny,frames],dtype=tf.float32)
        phi_j = tf.placeholder(shape=[None,Nx,Ny,frames],dtype=tf.float32)
        a_i   = tf.placeholder(shape=[None,1],dtype=tf.uint8)
        r_i   = tf.placeholder(shape=[None,1],dtype=tf.float32)
        t_i   = tf.placeholder(shape=[None,1],dtype=tf.float32)
        return phi_i,phi_j,a_i,r_i,t_i

    def get_initializers(self):
        graph_init = tf.global_variables_initializer()
        graph_local_init = tf.local_variables_initializer()
        return graph_init,graph_local_init

    def q_i_online(self,phi_i):
        with tf.name_scope('Qi_online'):
            q = self.q_net.run(phi_i,'online',True,False)
        return q

    def q_j_online(self,phi_j):
        with tf.name_scope('Qj_online'):
            q = self.q_net.run(phi_j,'online',True,True)
        return q

    def q_j_target(self,phi_j):
        with tf.name_scope('Qj_target'):
            q_j = self.q_net.run(phi_j,'target',False,False)
        return q_j

    def value_function_i_online(self,a_i,q_i,output_dimension):
        with tf.name_scope('Value_function_i_online'):
            # convert actions that were taken into onehot format
            a_list = tf.reshape(tf.cast(a_i,tf.int32),[-1])
            a_onehot = tf.one_hot(a_list, output_dimension)

            # now use the onehot format actions to select the Q_i's that are actually
            # obtained by taking action a_i. n.b Qnet returns a value for Q for all actions
            # but we only want to know Q for the action taken
            v_i_tmp = tf.multiply(a_onehot,q_i)
            v_i = tf.reduce_sum(v_i_tmp, axis=1)
        return v_i

    def discounted_reward(self,r_i,t_i,v_j):
        with tf.name_scope('discounted_reward'):
            discounted_r = tf.add( tf.squeeze(r_i) ,
                                  self.discount_factor
                                  *tf.multiply(tf.squeeze(t_i),
                                               tf.squeeze(v_j)))
        return discounted_r

    def get_loss(self,discounted_r, v_i):
        with tf.name_scope('discount_take_value'):
            difference = tf.subtract( discounted_r, v_i  )
        loss=self._loss(difference)
        return loss

    def _loss(self,offset):
        with tf.name_scope('loss'):
            loss_scale = 2.0 #self.HYPERPARAMS['LOSS_SCALE'] # how steep loss is for large values
            loss = tf.reduce_mean( loss_scale*(
                                    tf.sqrt(1.0+(1.0/loss_scale)**2
                                    *tf.multiply(offset,offset))
                                     - 1.0      )
                                  )
        return loss

    def get_train_op(self,loss):
        with tf.name_scope('optimizer'):
            optimizer    = tf.train.RMSPropOptimizer(self.learning_rate)
            train_op     = optimizer.minimize(loss)
        return train_op

    def make(self,graph,Nx,Ny,frames,output_dimension):
        # if( any(i is None for i in [self.N_x,self.N_y,self.frames] )):
        #     raise RuntimeError("call TrainingGraphFactory.setup()")
        with graph.as_default():
            phi_i,phi_j,a_i,r_i,t_i = self.set_placeholders(Nx,Ny,frames)
            Q_i = self.q_i_online(phi_i)
            V_i = self.value_function_i_online(a_i,Q_i,output_dimension)
            V_j = self.value_j_online(phi_j,output_dimension)
            discounted_r = self.discounted_reward(r_i,t_i,V_j)
            loss = self.get_loss(discounted_r,V_i)
            train_op = self.get_train_op(loss)
            update_target = self.update_all_layers()

            graph_init,graph_local_init = self.get_initializers()

            saver = tf.train.Saver()

            graph_vars = {'graph_init':graph_init,
                            'graph_local_init':graph_local_init,
                            'Q_i':Q_i,
                            'loss':loss,
                            'train_op':train_op,
                            'update_target':update_target,
                            'phi_i':phi_i,
                            'phi_j':phi_j,
                            'a_i':a_i,
                            'r_i':r_i,
                            't_i':t_i,
                            'saver':saver}

            output_graph = TrainingGraph(graph_vars)
            return output_graph


class DDQNTrainingGraphFactory(TrainingGraphFactory):

    def value_j_online(self,phi_j,output_dimension):
        # we need to get the actions to take on the Q_target step, by using the expected action
        # that the online network predicts is best, but then use the Q from the target net with the
        # online selected action

        # this is the same network as for Q_i_ - we set reuse=True
        # (it is also trainable)
        with tf.name_scope('Qj_online_and_inds'):
            Qj_online = self.q_j_online(phi_j)
            Qj_online_inds = tf.argmax(Qj_online,axis=1)
            Qj_onehot_inds = tf.one_hot(Qj_online_inds, output_dimension)

        # this has reuse=False, make a new network - the target network
        # it is not trainable. Instead we train the online network and
        # set the weights/biases of the layers in the target network to be the
        # same as those in the online network every so many games.
        Q_j=self.q_j_target(phi_j)

        # now only take values of Q (target) for state j, using action that
        # the online network would predict
        with tf.name_scope('value_function_j'):
            v_j = tf.reduce_sum(tf.multiply(Qj_onehot_inds,Q_j),axis=1)
        return v_j
