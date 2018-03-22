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

    def __init__(self, pos=(0, 0), alpha=0.0001, epsilon_decay=0.9984, epsilon=0.5, tau=1.0, tau_inc=0.3, init_tau=1.0,
                 lamb=0.5, gamma=0.99):
        '''
        Create a new agent

        Parameters
        ----------
        pos: initial position, default to (0,0)
        '''
        self.position = pos  # current position of the agent throughout the learning
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.tau_inc = tau_inc
        self.init_tau = init_tau
        self.lamb = lamb

        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []

    def softmax(self, q):
        '''
        Softmax decision function

        Parameters
        ----------
        q: Q table

        Returns
        -------
        ?
        '''
        assert self.tau >= 0.0
        q_tilde = q
        factors = np.exp(self.tau * q_tilde)
        return factors / np.sum(factors)

    def act_with_softmax(self, s, q):
        '''
        Agent acts with softmax

        Parameters
        ----------
        s: current state
        q: Q table

        Returns
        -------
        ?
        '''

        prob_a = self.softmax(q[s, :])
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
        td = r + self.gamma * q[s_prime, a_prime] - q[s, a]  # delta_t = r + gamme * Q(s',a') - Q(s,a)
        return q[s, a] + self.alpha * td  # Q(s,a) = Q(s,a) + alpha * delta_t

    def q_learning_update(self, q, s, a, r, s_prime):
        '''
        :param q: the q_table for action value function
        :param s: old state
        :param a: old action
        :param r: reward
        :param s_prime: new state
        :return: the update score
        '''
        td = r + self.gamma * np.max(q[s_prime, :]) - q[s, a]  # delta_t = r + gamma * max[Q(s',a')] - Q(s,a)
        return q[s, a] + self.alpha * td  # Q(s,a) = Q(s,a) + alpha * delta_t

    def get_action(self, s_prime, q_table, good_acts = [], method="greedy"):
        '''
        Given an action according to the method

        :param s: current state
        :param q_table: action state value
        :param method: how to act
        :return: an action
        '''
        if len(good_acts) != 0:
            n_trashes_surrounding = len(good_acts)
            a = good_acts[0] if n_trashes_surrounding == 1 else good_acts[np.random.randint(n_trashes_surrounding)]
            return a
        if method == "greedy":
            return self.act_with_epsilon_greedy(s_prime, q_table)
        if method == "softmax":
            return self.act_with_softmax(s_prime, q_table)

    def update(self, q_table, s, a, reward, s_prime, a_prime = None, e_table = None):
        '''
        Update the q_table according to the method
        But method is implicitely given by the number of arguments

        :param q_table: action state value function
        :param s:  current state
        :param a: current action
        :param reward: current reward
        :param s_prime: next state
        :param a_prime: next action
        :param e_table: eligibility table
        :return: None
        '''
        if e_table is None:
            if a_prime is None:             # Q_learning
                q_table[s, a] = self.q_learning_update(q_table, s, a, reward, s_prime)
            else:                           # Sarsa
                q_table[s, a] = self.sarsa_update(q_table, s, a, reward, s_prime, a_prime)
        else:                           # Eligibility trace
            e_table[s, a] = 1           # update eligibility trace when s = s_t | replacing trace
            if a_prime is None:
                delta = self.q_learning_update(q_table, s, a, reward, s_prime)
            else:
                delta = self.sarsa_update(q_table, s, a, reward, s_prime, a_prime)
            (n_s, n_a) = q_table.shape
            for u in range(n_s):
                for b in range(n_a):
                    q_table[u, b] = q_table[u, b] + self.alpha * delta * e_table[u, b]  # Q(s,a) = Q(s,a) + alpha * delta_t * e(s,a)
                e_table[u, :] = self.gamma * self.lamb * e_table[u, :]  # e(s,a) = gamma * e(s,a)

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def discounted_return(self, t = 0):
        H = len(self.ep_as)  # get the horizon
        discounted_returns = np.zeros(H - t)
        current_discounted_return = 0
        for tprime in reversed(range(t, H)):
            current_discounted_return = current_discounted_return * self.gamma + self.ep_rs[tprime]
            discounted_returns[tprime - t] = current_discounted_return
        return discounted_returns