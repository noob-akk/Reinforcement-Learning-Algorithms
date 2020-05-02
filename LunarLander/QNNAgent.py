import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import random
from keras.callbacks import LearningRateScheduler

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class QNNAgent(object):
	"""docstring for QNNAgent"""

	QNN = None
	state_space_dim = 0
	action_space_dim = 0

	def __init__(self, state_space_dim, action_space_dim):
		self.QNN = DiscreteActionQNN(state_space_dim, action_space_dim)
		self.action_space_dim = action_space_dim
		self.state_space_dim = state_space_dim

	def action_value(self, state, action):
		action_vec = [0]*self.action_space_dim
		action_vec[action] = 1
		return self.QNN.predict([list(state) + action_vec])[0,0]

	def action_values(self, state):
		state = list(state) 
		action_values = [0.]*self.action_space_dim
		for action in range(self.action_space_dim):
			action_values[action] = self.action_value(state, action)
		return action_values

	def best_action(self, state):
		return np.argmax(self.action_values(state))

	def eps_greedy_action(self, state, epsilon=0.5):
		rv = np.random.uniform(low=0., high=1., size=1)[0]
		if rv < (1 - epsilon):
			return self.best_action(state)
		else:
			return int((rv - (1 - epsilon))/epsilon*self.action_space_dim)

	def train(self, X, Y, epochs=1, lr=1e-3):
		self.QNN.lr = lr
		return self.QNN.train(X, Y, epochs=epochs)

	def max_action_value(self, state):
		return np.max(self.action_values(state))


class DiscreteActionQNN(object):
	"""
	Discrete Action Space QNN developed using a naive dense network
	"""

	Qmodel = None
	state_space_dim = 0
	action_space_dim = 0
	lr = 1e-3

	def __init__(self, state_space_dim, action_space_dim):
		self.Qmodel = self._naiveDenseNetwork(state_space_dim, action_space_dim, show_summary=True)
		self.action_space_dim = action_space_dim
		self.state_space_dim = state_space_dim

	def _naiveDenseNetwork(self, state_space_dim, action_space_dim, show_summary=False):
		Qmodel = Sequential()
		input_shape = state_space_dim+action_space_dim
		Qmodel.add(Dense(int(input_shape*1.75), activation='relu', input_shape=(input_shape,)))
		Qmodel.add(Dense(int(input_shape*1.2), activation='relu'))
		Qmodel.add(Dense(int(input_shape*0.5), activation='relu'))
		Qmodel.add(Dense(1, activation='linear'))
		Qmodel.compile("adam", loss="mse")
		Qmodel.summary()
		return Qmodel

	# This is a sample of a scheduler I used in the past
	def lr_scheduler(self, epoch, lr):
	    return self.lr

	def train(self, X, Y, epochs=1):
		if len(Y)>100:
			batch_size=64
		elif len(Y)>50:
			batch_size=32
		else:
			batch_size=16
		combined = list(zip(X, Y))
		random.shuffle(combined)
		X[:], Y[:] = zip(*combined)
		callbacks = [LearningRateScheduler(self.lr_scheduler, verbose=1)]
		hist = self.Qmodel.fit(X, Y, epochs=epochs, callbacks=callbacks, batch_size=batch_size, verbose=0)
		return hist

	def predict(self, X):
		return self.Qmodel.predict([X], verbose=2)