import tensorflow as tf
from tensorflow import keras
import numpy as np
from abc import ABC,abstractmethod

class Qnet(ABC):
    def __init__(self,N_x=1,N_y=1,frames=1):
        self.N_x=N_x
        self.N_y=N_y
        self.frames=frames
        pass

    def set_dimensions(self,N_x,N_y,frames,output_dimension):
        self.N_x=N_x
        self.N_y=N_y
        self.frames=frames
        self.output_dimension=output_dimension

    @abstractmethod
    def run(self):
        pass

class TripleConvQnet(Qnet):

    def __init__(self,conv_filters=2,fc_units=32):
        super().__init__()
        self.conv_filters=conv_filters
        self.fc_units=fc_units

    def set_dimensions(self,N_x,N_y,frames,output_dimension):
        super().set_dimensions(N_x,N_y,frames,output_dimension)
        o1 = int( ( (self.N_x-8)/4 ) + 1 )
        o2 = int( ( (o1-4)/2 ) + 1 )
        o3 = int( ( (o2-3)/1) + 1)
        self.conv_squash_factor = o3

    def run(self,obs,call_type,trainme=True,reuseme=False):
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

        Keyword Args:
            call_type:
            trainme:

        Returns:
            z_out: output of the Neural Net, which is the predicted Q for the observation

        """

        with tf.variable_scope(call_type):
            z = tf.reshape(obs, [-1,self.N_x,self.N_y,self.frames])
            #print(z.shape)
            with tf.variable_scope('conv_layer0',reuse=reuseme):
                z_conv0 = tf.layers.Conv2D(filters = self.conv_filters,
                                            kernel_size = (8,8),
                                            strides = (4,4),
                                            padding='valid',
                                            activation=tf.nn.leaky_relu,
                                            trainable=trainme,
                                            kernel_initializer=tf.keras.initializers.he_normal())(z)

            with tf.variable_scope('conv_layer1',reuse=reuseme):
                z_conv1 = tf.layers.Conv2D(filters = 2*self.conv_filters,
                                            kernel_size = (4,4),
                                            strides = (2,2),
                                            padding='valid',
                                            activation=tf.nn.leaky_relu,
                                            trainable=trainme,
                                            kernel_initializer=tf.keras.initializers.he_normal())(z_conv0)
                #z_conv1_flat = tf.reshape(z_conv1,[-1,self.PARAMS['N_squash']*self.PARAMS['N_squash']*(2*self.HYPERPARAMS['N_FILTER'])])

            with tf.variable_scope('conv_layer2',reuse=reuseme):
                z_conv2 = tf.layers.Conv2D(filters = 2*self.conv_filters,
                                            kernel_size = (3,3),
                                            strides = (1,1),
                                            padding='valid',
                                            activation=tf.nn.leaky_relu,
                                            trainable=trainme,
                                            kernel_initializer=tf.keras.initializers.he_normal())(z_conv1)
                z_flat = tf.reshape(z_conv2,[-1,self.conv_squash_factor**2*(2*self.conv_filters)])


            with tf.variable_scope('FC_layer0',reuse=reuseme):
                z_FC0 =  tf.layers.Dense(units=self.fc_units,activation=tf.nn.relu,trainable=trainme,kernel_initializer=tf.keras.initializers.he_normal())(z_flat)

            with tf.variable_scope('layer_out',reuse=reuseme):
                z_out = tf.layers.Dense(units=self.output_dimension,trainable=trainme,kernel_initializer=tf.keras.initializers.he_normal())(z_FC0)

        return z_out
