import numpy as np
from enum import IntEnum

# ====== ACTION SPACE =====#

class Action(IntEnum):
	'''Class that represents the action space'''
	LEFT = 0
	RIGHT = 1
	DOWN = 2
	UP = 3

	@classmethod
	def size(self):
		'''
		Return the size of the action space
		'''
		return len(self)



# ====== AGENT ====== #

class Agent:
	'''Class that represents our hardworking agent'''

	def __init__(self, pos=(0,0), alpha=0.1, decay_factor=0.92, epsilon=0.6, tau=1.0, tau_inc=0.01, init_tau=1.0, lamb=0.9, gamma=0.99):
		'''
		Create a new agent

		Parameters
		----------
		pos: initial position, default to (0,0)
		'''
		self.position = pos # current position of the agent throughout the learning
		self.alpha = alpha
		self.decay_factor = decay_factor
		self.epsilon = epsilon
		self.gamma = gamma
		self.tau = tau
		self.tau_inc = tau_inc
		self.init_tau = init_tau
		self.lamb = lamb


	def softmax(self, q):
		'''
		Softmax decision function

		Parameters
		----------
		'''
		assert self.tau >= 0.0
		q_tilde = q - np.max(q)
		factors = np.exp(self.tau * q_tilde)
		return factors / np.sum(factors)

	
	def act_with_softmax(self, s, q):
		'''Agent acts with softmax'''
		
		prob_a = softmax(q[s, :])
		cumsum_a = np.cumsum(prob_a)
		return np.where(np.random.rand() < cumsum_a)[0][0]


	def act_with_epsilon_greedy(self, s, q):
		'''
		take an action according to the epsilon_greedy policy

		:param s: the state of the environment
		:param q: the q_table for action value function
		:return: the action decided by epsilon greedy
		'''
		a = np.argmax(q[s, :])
		if np.random.rand() < self.epsilon:
			a = np.random.randint(q.shape[1])
		return a


	def sarsa_update(self, q, s, a, r, s_prime, a_prime):
		'''
		:param q: the q_table for action value function
		:param s: old state
		:param a: old action
		:param r: reward
		:param s_prime: new state
		:param a_prime: new action
		:return: the update score
		'''
		td = r + self.gamma * q[s_prime, a_prime] - q[s, a]
		return q[s, a] + self.alpha * td


	def q_learning_update(self, q, s, a, r, s_prime):
		'''
		:param q: the q_table for action value function
		:param s: old state
		:param a: old action
		:param r: reward
		:param s_prime: new state
		:return: the update score
		'''
		td = r + self.gamma * np.max(q[s_prime, :]) - q[s, a]
		return q[s, a] + self.alpha * td