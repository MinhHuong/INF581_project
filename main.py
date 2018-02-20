from environment import Environment
from enum import IntEnum
import random
import numpy as np
import matplotlib.pyplot as plt

class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3
    # I am still concerned abt this action version #


"""
obs, r, flag, info = env.step(Action.UP)
print(info)
print(r)
env.display()
obs, r, flag, info = env.step(Action.DOWN)
print(info)
print(r)
env.display()
obs, r, flag, info = env.step(Action.DOWN)
print(info)
print(r)
env.display()
obs, r, flag, info = env.step(Action.RIGHT)
print(info)
print(r)
env.display()
obs, r, flag, info = env.step(Action.RIGHT)
print(info)
print(r)
env.display()
"""


#print("Nb_Actions: {}".format(n_a))
#print("Nb_States: {}".format(n_s))

'''
print("reset test")
env.reset()
env.display()

print("reset test")
env.reset()
env.display()
'''

# parameters for the RL agent
alpha = 0.1
decay_factor = 0.92
epsilon = 0.6
gamma = 0.99
tau = 1
tau_inc = 0.01
init_tau = 1
lamb = 0.9

def softmax(q):
    assert tau >= 0.0
    q_tilde = q - np.max(q)
    factors = np.exp(tau * q_tilde)
    return factors / np.sum(factors)

def act_with_softmax(s, q):
    prob_a = softmax(q[s, :])
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]

def act_with_epsilon_greedy(s, q):
    '''
    take an action according to the epsilon_greedy policy

    :param s: the state of the environment
    :param q: the q_table for action value function
    :return:
    '''
    a = np.argmax(q[s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q.shape[1])
    #print(Action(a))
    return a

def sarsa_update(q, s, a, r, s_prime, a_prime):
    '''
    :param q: the q_table for action value function
    :param s: old state
    :param a: old action
    :param r: reward
    :param s_prime: new state
    :param a_prime: new action
    :return: the update score
    '''
    td = r + gamma * q[s_prime, a_prime] - q[s, a]
    return q[s, a] + alpha * td

def q_learning_update(q, s, a, r, s_prime):
    '''
    :param q: the q_table for action value function
    :param s: old state
    :param a: old action
    :param r: reward
    :param s_prime: new state
    :return: the update score
    '''
    td = r + gamma * np.max(q[s_prime, :]) - q[s, a]
    return q[s, a] + alpha * td


if __name__ == '__main__':

    env = Environment()

    #env.display()

    n_a = env.action_space_n
    n_s = env.state_space_n

    q_table = np.zeros([n_s, n_a])
    e_table = np.zeros([n_s, n_a])

    for i_episode in range(20):
        s = env.reset()
        print("episode {}".format(i_episode))
        a = act_with_epsilon_greedy(s, q_table)
        #a = act_with_softmax(s, q_table)
        total_return = 0.0

        for t in range(300):

            # Act
            s_prime, reward, done, info = env.step(a)

            total_return += np.power(gamma, t) * reward
            if (i_episode > 18):
                env.display()
                print(info)

            # Select an action
            #a_prime = act_with_softmax(s, q_table)
            a_prime = act_with_epsilon_greedy(s_prime, q_table)

            # update a Q value table
            delta = sarsa_update(q_table, s, a, reward, s_prime, a_prime)
            #q_table[s, a] = q_learning_update(q_table, s, a, reward, s_prime)
            e_table[s, a] = e_table[s, a] + 1

            # Update q_table and e_table
            for u in range(n_s):
                for b in range(n_a):
                    q_table[u, b] = q_table[u, b] + alpha * delta * e_table[u, b]
                e_table[u] = gamma * lamb * e_table[u]
            e_table[s] = e_table[s] / gamma / lamb

            # Transition to new state
            s = s_prime
            a = a_prime

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print(info)
                break
        epsilon = epsilon * decay_factor
        tau = init_tau + i_episode * tau_inc
        print("total return {}".format(total_return))
        print("percentage of cleaning {}".format((30 - len(env.trashes))/30))
        #print("epsilon {}".format(epsilon))

    #print(q_table)
    #env.display()
    np.savetxt('q_table.dat', q_table, fmt='%f')
