""" DQN package for deepQ learning

This module provides methods for applying deep Q learning to atari games, with a
focus on the game 'pong'

This package contains two classes:

Qmemory: for storing and retrieving 'experience memory' of an agent, and
deepQ: which contains a series of methods for trainingm and testing agents

"""
from DQN.qmemory import Qmemory
from DQN.deepq import deepQ
