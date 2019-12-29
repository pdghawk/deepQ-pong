
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class TraningGraph:
    def __init__(self,graph_vars):
        self.graph_vars=graph_vars
        self._check_expected()

    def _check_expected(self):
        expected=[
        'graph_init',
        'graph_local_init',
        'Q_i',
        'loss',
        'train_op':,
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
                raise ValueError("make sure all appropriate fields are filled \
                                  in input to TrainingGraph"))


    def initialize_graph(self):
        sess.run(self.graph_vars['graph_init'])
        sess.run(self.graph_vars['graph_local_init'])

    def construct_feed_dict(self,phi_i,phi_j,a_i,r_i,t_i):
        feed_dict = {self.graph_vars['phi_i']:phi_i,
                    self.graph_vars['phi_j']:phi_j,
                    self.graph_vars['a_i']:a_i,
                    self.graph_vars['r_i']:r_i,
                    self.graph_vars['t_i']:t_i}
        return feed_dict

    def get_q(self,session,phi_i,phi_j,a_i,r_i,t_i):
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
