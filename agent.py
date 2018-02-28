import numpy as np
from enum import IntEnum
from sklearn.neural_network import MLPRegressor
#from xgboost import XGBRegressor

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

    def __init__(self, pos=(0, 0), alpha=0.0001, epsilon_decay=0.999, epsilon=0.5, tau=1.0, tau_inc=0.01, init_tau=1.0,
                 lamb=0.3, gamma=0.9):
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
        q_tilde = q - np.max(q)
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

    # not correctly implemented!!!
    def Policy_gradient(self, policy, env, max_iter):
        '''
        implementation of policy gradient to find the best policy

        :param policy: initial policy
        :param env: environment
        :param max_iter: maximum iterations per episode
        :return: policy
        '''

        alpha = 0.1

        for iter in range(500): # let's say 500

            ## do rollout following softmax exploration with policy as a params
            s_t = env.reset()
            states, actions, rewards = [], [], []
            states.append(s_t)
            for t in range(max_iter):
                a_t = self.act_with_softmax(s_t, policy)
                s_t, r_t, done, info = env.step(a_t)
                states.append(s_t)
                actions.append(a_t)
                rewards.append(r_t)
                if done:
                    break

            ## policy gradient
            H = len(rewards)        # Horizon
            PG = 0                  # Policy gradient
            for t in range(H):
                pi = self.softmax(policy[states[t], :])
                R_t = sum(rewards[t::])
                g_Qtable_log_pi = (1 - pi) * R_t * (1./self.tau)
                PG += g_Qtable_log_pi
            policy[s_t, :] += alpha * PG
            self.tau = self.init_tau + iter * self.tau_inc
        return policy

    # do not whether it is well implemented yet
    def NFQ(self, env, max_iter=100, n_timesteps=100):
        '''
        implementation of neural fitted Q iteration

        :param env: environment
        :param max_iter: maximum iterations per episode
        :return: q_table
        '''

        n_s = env.state_space_n
        n_a = env.action_space_n

        q_table = np.random.randn(n_s, n_a) # initial q_table
        XX = []                             # to store data
        q_target = []                       # to store target

        for iter in range(max_iter):

            '''
            TODO:
            replace sklearn package with tensorflow + keras
            like trying RNN or CNN
            https://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/
            https://keras.io/layers/recurrent/
            Also, we can play with the structure of learner below
            '''
            learner = MLPRegressor(activation="relu",
                                   hidden_layer_sizes=(100, 3),
                                   max_iter=2000,
                                   solver='lbfgs')
            '''
            learner = XGBRegressor(n_estimators=500,
                                   max_depth=4,
                                   learning_rate=0.07,
                                   subsample=.9,
                                   min_child_weight=6,
                                   colsample_bytree=.8,
                                   scale_pos_weight=1.6,
                                   gamma=10,
                                   reg_alpha=8,
                                   reg_lambda=1.3)
            '''
            ## do rollout following softmax policy
            s_t = env.reset()
            for t in range(n_timesteps):
                a_t = self.act_with_epsilon_greedy(s_t, q_table)
                s_tprime, r_t, done, info = env.step(a_t)
                XX.append([s_t, a_t])
                q_target.append(r_t + self.gamma * max(q_table[s_tprime, :]))
                if done:
                    break
                s_t = s_tprime

            self.epsilon = self.epsilon * self.epsilon_decay
            self.tau = self.init_tau + iter * self.tau_inc

            # Regression
            learner.fit(XX, np.array(q_target))

            # Resample
            for s in range(n_s):
                    q_table[s, :] = learner.predict([[s, a] for a in range(n_a)])
        #print(q_table)
        return q_table

    def get_action(self, s_prime, q_table, method="greedy"):
        '''
        Given an action according to the method

        :param s: current state
        :param q_table: action state value
        :param method: how to act
        :return: an action
        '''
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
        if a_prime is None:             # Q_learning
            q_table[s, a] = self.q_learning_update(q_table, s, a, reward, s_prime)
        elif e_table is None:           # Sarsa
            q_table[s, a] = self.sarsa_update(q_table, s, a, reward, s_prime, a_prime)
        else:                           # Eligibility trace
            e_table[s, a] = 1           # update eligibility trace when s = s_t | replacing trace
            delta = self.sarsa_update(q_table, s, a, reward, s_prime, a_prime)
            (n_s, n_a) = q_table.shape
            for u in range(n_s):
                for b in range(n_a):
                    q_table[u, b] = q_table[u, b] + self.alpha * delta * e_table[u, b]  # Q(s,a) = Q(s,a) + alpha * delta_t * e(s,a)
                e_table[u, :] = self.gamma * self.lamb * e_table[u, :]  # e(s,a) = gamma * e(s,a)
            e_table[s, a] = e_table[s, a] / self.gamma / self.lamb  # ??? :-( je comprends pas